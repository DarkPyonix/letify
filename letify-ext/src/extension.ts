/**
 * Extension entry point: status bar items, hover cards, the letify view and polling.
 *
 * Owns the VS Code wiring. Data comes only from the letify CLI's JSON (spec "Editor
 * extension"); formatting lives in model.ts and render.ts, history in history.ts.
 */

import * as vscode from "vscode";

import { runLetify } from "./cli";
import { DayBucket, recordSample } from "./history";
import {
  Status,
  UsageRow,
  UtilizationRow,
  gpuStatusText,
  gpuSummary,
  mostConstrained,
  parseStatus,
  parseUsage,
  parseUtilization,
  quotaStatusText,
  severity,
  shareLeft,
  escapeHtml,
} from "./model";
import { STYLE, gpuCard, historyChart, quotaCard, runtimesCard } from "./render";

const HISTORY_KEY = "letify.history";

interface State {
  usage: UsageRow[];
  utilization: UtilizationRow[];
  status: Status | null;
  errors: Record<string, string>;
  asOf: number | null;
}

type Tab = "quota" | "gpu" | "runtimes";

export function activate(context: vscode.ExtensionContext): void {
  const state: State = { usage: [], utilization: [], status: null, errors: {}, asOf: null };
  const quotaItem = vscode.window.createStatusBarItem("letify.quota", vscode.StatusBarAlignment.Right, 100);
  const gpuItem = vscode.window.createStatusBarItem("letify.gpu", vscode.StatusBarAlignment.Right, 99);
  quotaItem.name = "letify quota";
  gpuItem.name = "letify GPU activity";
  quotaItem.command = { title: "Show quota", command: "letify.show", arguments: ["quota"] };
  gpuItem.command = { title: "Show GPU activity", command: "letify.show", arguments: ["gpu"] };

  let view: vscode.WebviewView | undefined;
  let tab: Tab = "quota";
  const config = () => vscode.workspace.getConfiguration("letify");
  const cwd = () => vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  const history = (): DayBucket[] => context.globalState.get<DayBucket[]>(HISTORY_KEY, []);

  const read = async (subcommand: string): Promise<unknown | undefined> => {
    try {
      const data = await runLetify(config().get<string>("command", "uv run letify"), subcommand, cwd());
      delete state.errors[subcommand];
      return data;
    } catch (error) {
      state.errors[subcommand] = error instanceof Error ? error.message : String(error);
      return undefined;
    }
  };

  const tooltip = (body: string): vscode.MarkdownString => {
    const md = new vscode.MarkdownString(body);
    md.isTrusted = { enabledCommands: ["letify.refresh", "letify.show", "letify.openSettings"] };
    md.supportHtml = true;
    md.supportThemeIcons = true;
    return md;
  };

  const footer = (markdown: boolean): string => {
    const time = state.asOf ? new Date(state.asOf * 1000).toLocaleTimeString() : "never";
    const errors = Object.values(state.errors).map((m) => escapeHtml(m));
    if (markdown) {
      const lines = errors.map((m) => `$(error) ${m}`);
      lines.push(`From ${escapeHtml(state.status?.name || "this workspace")} · as of ${time} · [Refresh](command:letify.refresh) · [Settings](command:letify.openSettings)`);
      return lines.join("\n\n");
    }
    const errorHtml = errors.map((m) => `<div class="card error">${m}</div>`).join("");
    return `${errorHtml}<div class="footer muted"><span>From ${escapeHtml(state.status?.name || "this workspace")} · as of ${time}</span><span><a data-cmd="refresh">Refresh</a> · <a data-cmd="settings">Settings</a></span></div>`;
  };

  const plainLines = {
    quota: (now: number) =>
      state.usage.map((row) => {
        const left = shareLeft(row);
        if (row.unavailable) return `**${escapeHtml(row.alias)}** unavailable`;
        if (row.unmetered) return `**${escapeHtml(row.alias)}** no quota`;
        return `**${escapeHtml(row.alias)}** ${left === null ? "" : `${Math.round((1 - left) * 100)}% used · `}${escapeHtml(quotaStatusText([row], now))}`;
      }),
  };

  const renderStatusBar = () => {
    const now = Date.now() / 1000;
    const cfg = config();
    const worst = mostConstrained(state.usage);
    const level = severity(worst ? shareLeft(worst) : null, cfg.get("warningPercent", 20), cfg.get("errorPercent", 5));
    quotaItem.text = `$(pulse) ${state.errors.usage && state.usage.length === 0 ? "letify error" : quotaStatusText(state.usage, now)}`;
    quotaItem.backgroundColor =
      level === "error"
        ? new vscode.ThemeColor("statusBarItem.errorBackground")
        : level === "warning"
          ? new vscode.ThemeColor("statusBarItem.warningBackground")
          : undefined;
    quotaItem.tooltip = tooltip(
      [
        `**letify quota** · [Quota](command:letify.show?${encodeURIComponent('["quota"]')}) · [GPU](command:letify.show?${encodeURIComponent('["gpu"]')}) · [Runtimes](command:letify.show?${encodeURIComponent('["runtimes"]')})`,
        ...plainLines.quota(now),
        footer(true),
      ].join("\n\n"),
    );
    const summary = gpuSummary(state.utilization, state.status, cfg.get("busyPercent", 10));
    gpuItem.text = `$(server) ${gpuStatusText(summary)}`;
    const deviceLines = state.utilization.flatMap((row) =>
      row.devices.length
        ? row.devices.map((d) => `**${escapeHtml(row.alias)}** gpu${d.index} ${escapeHtml(d.name)} · ${d.utilization_percent ?? "?"}% · ${d.memory_percent === null ? "?" : Math.round(d.memory_percent)}% mem`)
        : [`**${escapeHtml(row.alias)}.${escapeHtml(row.accelerator)}** ${escapeHtml(row.reason ?? row.unavailable ?? "")}`],
    );
    gpuItem.tooltip = tooltip([`**letify GPU activity** · ${summary.reserved} reserved by letify`, ...deviceLines, footer(true)].join("\n\n"));
    cfg.get("showQuota", true) ? quotaItem.show() : quotaItem.hide();
    cfg.get("showGpu", true) ? gpuItem.show() : gpuItem.hide();
  };

  const renderView = () => {
    if (!view) return;
    const now = Date.now() / 1000;
    const tabs = (["quota", "gpu", "runtimes"] as Tab[])
      .map((t) => `<button class="${t === tab ? "active" : ""}" data-tab="${t}">${t === "quota" ? "Quota" : t === "gpu" ? "GPU" : "Runtimes"}</button>`)
      .join("");
    let body = "";
    if (tab === "quota") {
      body = (state.usage.length ? state.usage.map((r) => quotaCard(r, now)).join("") : `<div class="card muted">No usage read yet</div>`) + historyChart(history(), "quota");
    } else if (tab === "gpu") {
      body = (state.utilization.length ? state.utilization.map((r) => gpuCard(r, state.status)).join("") : `<div class="card muted">No accelerator declared, or nothing read yet</div>`) + historyChart(history(), "gpu");
    } else {
      body = runtimesCard(state.status);
    }
    const nonce = Math.random().toString(36).slice(2);
    view.webview.html = `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';">
<style>${STYLE}</style></head><body><div class="tabs">${tabs}</div>${body}${footer(false)}
<script nonce="${nonce}">const api=acquireVsCodeApi();document.addEventListener('click',ev=>{const t=ev.target.closest('[data-tab],[data-cmd]');if(!t)return;api.postMessage(t.dataset.tab?{tab:t.dataset.tab}:{cmd:t.dataset.cmd});});</script>
</body></html>`;
  };

  const render = () => {
    renderStatusBar();
    renderView();
  };

  const refreshUsage = async () => {
    const data = await read("usage");
    if (data !== undefined) {
      try {
        state.usage = parseUsage(data);
      } catch (error) {
        state.errors.usage = (error as Error).message;
      }
    }
  };

  const refreshGpu = async () => {
    const [util, status] = await Promise.all([read("utilization"), read("status")]);
    try {
      if (util !== undefined) state.utilization = parseUtilization(util);
      if (status !== undefined) state.status = parseStatus(status);
    } catch (error) {
      state.errors.utilization = (error as Error).message;
    }
  };

  const sample = async () => {
    const now = Date.now() / 1000;
    const remaining: Record<string, number> = {};
    for (const row of state.usage) if (row.remaining !== null) remaining[`${row.alias}:${row.unit}`] = row.remaining;
    const summary = gpuSummary(state.utilization, state.status, config().get("busyPercent", 10));
    await context.globalState.update(HISTORY_KEY, recordSample(history(), now, summary.meanPercent, remaining));
  };

  let refreshing = false;
  const refreshAll = async () => {
    if (refreshing) return;
    refreshing = true;
    try {
      await Promise.all([refreshUsage(), refreshGpu()]);
      state.asOf = Date.now() / 1000;
      await sample();
    } finally {
      refreshing = false;
      render();
    }
  };

  let usageTimer: NodeJS.Timeout | undefined;
  let gpuTimer: NodeJS.Timeout | undefined;
  const schedule = () => {
    if (usageTimer) clearInterval(usageTimer);
    if (gpuTimer) clearInterval(gpuTimer);
    const usageSeconds = Math.max(10, config().get<number>("usageIntervalSeconds", 60));
    const gpuSeconds = view?.visible ? Math.max(2, config().get<number>("utilizationIntervalSeconds", 10)) : usageSeconds;
    usageTimer = setInterval(async () => {
      await refreshUsage();
      state.asOf = Date.now() / 1000;
      await sample();
      render();
    }, usageSeconds * 1000);
    gpuTimer = setInterval(async () => {
      await refreshGpu();
      render();
    }, gpuSeconds * 1000);
  };

  context.subscriptions.push(
    quotaItem,
    gpuItem,
    { dispose: () => { if (usageTimer) clearInterval(usageTimer); if (gpuTimer) clearInterval(gpuTimer); } },
    vscode.window.registerWebviewViewProvider("letify.view", {
      resolveWebviewView(webviewView) {
        view = webviewView;
        webviewView.webview.options = { enableScripts: true };
        webviewView.webview.onDidReceiveMessage((message: { tab?: Tab; cmd?: string }) => {
          if (message.tab) {
            tab = message.tab;
            renderView();
          } else if (message.cmd === "refresh") {
            void refreshAll();
          } else if (message.cmd === "settings") {
            void vscode.commands.executeCommand("letify.openSettings");
          }
        });
        webviewView.onDidChangeVisibility(() => {
          schedule();
          if (webviewView.visible) void refreshGpu().then(render);
        });
        webviewView.onDidDispose(() => {
          view = undefined;
          schedule();
        });
        renderView();
        schedule();
      },
    }),
    vscode.commands.registerCommand("letify.refresh", () => refreshAll()),
    vscode.commands.registerCommand("letify.openSettings", () =>
      vscode.commands.executeCommand("workbench.action.openSettings", "@ext:letify.letify-status"),
    ),
    vscode.commands.registerCommand("letify.show", async (which?: Tab) => {
      if (which) tab = which;
      await vscode.commands.executeCommand("letify.view.focus");
      renderView();
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("letify")) {
        schedule();
        render();
      }
    }),
  );

  quotaItem.text = "$(sync~spin) letify";
  render();
  schedule();
  void refreshAll();
}

export function deactivate(): void {}
