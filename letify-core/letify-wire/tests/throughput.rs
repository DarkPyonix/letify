//! Loopback throughput of one large copy to the device.
//!
//! Ignored by default because it moves 256 MiB several times. Run it with
//! `cargo test --release -p letify-wire --test throughput -- --ignored --nocapture`.
//!
//! The client side mirrors what letify-driver does for `cuMemcpyHtoD_v2`: a `BufWriter`
//! over a TCP stream with `TCP_NODELAY` set. The agent side mirrors letify-agent: a
//! `BufReader` over the accepted stream. Only the path the bytes take through the wire
//! crate differs between the measurements.

use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Instant;

use letify_wire::{Request, decode_request, encode_request, read_frame, write_frame};

const PAYLOAD_BYTES: usize = 256 * 1024 * 1024;
const RUNS: usize = 5;

/// Run one transfer per run and return the median MiB/s.
fn measure<S, R>(send: S, receive: R) -> f64
where
    S: Fn(&mut BufWriter<TcpStream>, &[u8]) + Send + Sync + Copy + 'static,
    R: Fn(&mut BufReader<TcpStream>) -> usize + Send + Copy + 'static,
{
    let payload: &'static [u8] = Box::leak(vec![0xa5u8; PAYLOAD_BYTES].into_boxed_slice());
    let mut rates = Vec::new();
    for _ in 0..RUNS {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let agent = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            let mut reader = BufReader::new(stream);
            receive(&mut reader)
        });
        let stream = TcpStream::connect(address).unwrap();
        stream.set_nodelay(true).unwrap();
        let mut writer = BufWriter::new(stream);
        let started = Instant::now();
        send(&mut writer, payload);
        writer.flush().unwrap();
        let received = agent.join().unwrap();
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(received, PAYLOAD_BYTES);
        rates.push(PAYLOAD_BYTES as f64 / (1024.0 * 1024.0) / seconds);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rates[RUNS / 2]
}

#[test]
#[ignore]
fn a_256_mib_copy_to_the_device_through_an_encoded_frame() {
    let rate = measure(
        |writer, payload| {
            let request = Request::CopyToDevice { handle: 1, offset: 0, payload: payload.to_vec() };
            write_frame(writer, &encode_request(&request)).unwrap();
        },
        |reader| {
            let frame = read_frame(reader).unwrap();
            match decode_request(&frame).unwrap() {
                Request::CopyToDevice { payload, .. } => payload.len(),
                other => panic!("unexpected {other:?}"),
            }
        },
    );
    println!("encoded frame: {rate:.0} MiB/s for a 256 MiB copy to the device");
}
