//! A stream of a session as a tokio `AsyncRead` and `AsyncWrite`. It owns the per-stream flow
//! control window on the read and write path. Ordering and retransmission belong to the session.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

use crate::error::Error;
use crate::frame::{self, CHUNK};
use crate::session::Guard;

/// One ordered byte stream inside a session. Closing it with `shutdown` ends the stream and
/// leaves the session open.
pub struct Stream {
    guard: Arc<Guard>,
    id: u32,
}

impl Stream {
    pub(crate) fn new(guard: Arc<Guard>, id: u32) -> Stream {
        Stream { guard, id }
    }

    /// The stream id. Odd ids are opened by the initiator and even ids by the responder.
    pub fn id(&self) -> u32 {
        self.id
    }
}

fn lost(why: &str) -> io::Error {
    Error::ConnectionLost(why.to_string()).into()
}

impl AsyncRead for Stream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let mut st = self.guard.shared.lock();
        let id = self.id;
        let (remote_closed, failed) = (st.remote_closed, st.failed.clone());
        let Some(s) = st.streams.get_mut(&id) else {
            return Poll::Ready(Ok(()));
        };
        if let Some((buf, off)) = s.recv_buf.front_mut() {
            let n = (buf.len() - *off).min(out.remaining());
            out.put_slice(&buf[*off..*off + n]);
            *off += n;
            if *off == buf.len() {
                s.recv_buf.pop_front();
            }
            st.consumed(id, n);
            return Poll::Ready(Ok(()));
        }
        if s.remote_fin || remote_closed {
            return Poll::Ready(Ok(()));
        }
        if let Some(why) = failed {
            return Poll::Ready(Err(lost(&why)));
        }
        s.read_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl AsyncWrite for Stream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let mut st = self.guard.shared.lock();
        let id = self.id;
        if let Some(why) = st.failed.clone() {
            return Poll::Ready(Err(lost(&why)));
        }
        let closing = st.closing;
        let Some(s) = st.streams.get(&id) else {
            return Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()));
        };
        if s.fin_sent || closing {
            return Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()));
        }
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let credit = s.send_credit;
        let n = buf
            .len()
            .min(CHUNK)
            .min(credit.min(usize::MAX as u64) as usize);
        if n == 0 || !st.has_room(n) {
            if let Some(s) = st.streams.get_mut(&id) {
                s.write_waker = Some(cx.waker().clone());
            }
            return Poll::Pending;
        }
        if let Some(s) = st.streams.get_mut(&id) {
            s.send_credit -= n as u64;
        }
        st.enqueue(|seq| frame::encode_data(seq, id, &buf[..n]));
        Poll::Ready(Ok(n))
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut st = self.guard.shared.lock();
        let id = self.id;
        if let Some(why) = st.failed.clone() {
            return Poll::Ready(Err(lost(&why)));
        }
        let send = match st.streams.get_mut(&id) {
            Some(s) if !s.fin_sent => {
                s.fin_sent = true;
                true
            }
            _ => false,
        };
        if send {
            st.enqueue_fin(id);
        }
        st.forget_if_done(id);
        Poll::Ready(Ok(()))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let mut st = self.guard.shared.lock();
        let id = self.id;
        let Some(s) = st.streams.get_mut(&id) else {
            return;
        };
        s.dropped = true;
        let unread: usize = s.recv_buf.drain(..).map(|(b, off)| b.len() - off).sum();
        let send_fin = !s.fin_sent && st.failed.is_none() && !st.closing;
        if send_fin {
            if let Some(s) = st.streams.get_mut(&id) {
                s.fin_sent = true;
            }
            st.enqueue_fin(id);
        }
        st.consumed(id, unread);
        st.forget_if_done(id);
    }
}
