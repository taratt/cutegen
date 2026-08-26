#!/usr/bin/env python3
"""Generate error-report canvas TSX from error payload JSON."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CANVAS_DIR = Path.home() / ".cursor/projects/home-tarasaba-PycharmProjects-cutegen-test/canvases"
SCRIPTS = Path(__file__).resolve().parent


def js(obj, indent=0):
    sp = "  " * indent
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, str):
        return json.dumps(obj)
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, (int, float, bool)) or x is None for x in obj):
            inner = ", ".join("null" if x is None else js(x) for x in obj)
            return f"[{inner}]"
        items = ",\n".join(sp + "  " + js(x, indent + 1) for x in obj)
        return f"[\n{items}\n{sp}]"
    if isinstance(obj, dict):
        items = ",\n".join(f'{sp}  {json.dumps(k)}: {js(v, indent + 1)}' for k, v in obj.items())
        return f"{{\n{items}\n{sp}}}"
    return json.dumps(obj)


ERROR_REPORT_BODY = '''
type KernelMeta = { id: number; name: string; ktype: string; cohort: string };
type ExpMeta = { col: string; path: string; label: string; backend: string; method: string; model: string; leaf: string };

type Cohort = "sample19" | "new8" | "all";
type ViewMode = "summary" | "kernels" | "compare";

function cohortIds(cohort: Cohort): number[] {
  if (cohort === "sample19") return SAMPLE19_IDS;
  if (cohort === "new8") return NEW8_IDS;
  return [...SAMPLE19_IDS, ...NEW8_IDS];
}

function kernelMeta(id: number): KernelMeta | undefined {
  return KERNELS.find((x) => x.id === id);
}

function kernelLabel(id: number): string {
  const k = kernelMeta(id);
  return k ? `${id} · ${k.ktype}` : String(id);
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function emptyErr(): KernelErr {
  return {
    nodes: 0,
    nodes_pass: 0,
    nodes_fail: 0,
    compile_errors: 0,
    correct_errors: 0,
    compile_resolved: 0,
    correct_resolved: 0,
    codegen_regens: 0,
    timeouts: 0,
    resolved: 0,
    unresolved: 0,
    max_depth: -1,
    ever_passed: false,
  };
}

function mergeErr(a: KernelErr, b: KernelErr): KernelErr {
  return {
    nodes: a.nodes + b.nodes,
    nodes_pass: a.nodes_pass + b.nodes_pass,
    nodes_fail: a.nodes_fail + b.nodes_fail,
    compile_errors: a.compile_errors + b.compile_errors,
    correct_errors: a.correct_errors + b.correct_errors,
    compile_resolved: a.compile_resolved + b.compile_resolved,
    correct_resolved: a.correct_resolved + b.correct_resolved,
    codegen_regens: a.codegen_regens + b.codegen_regens,
    timeouts: a.timeouts + b.timeouts,
    resolved: a.resolved + b.resolved,
    unresolved: a.unresolved + b.unresolved,
    max_depth: Math.max(a.max_depth, b.max_depth),
    ever_passed: a.ever_passed || b.ever_passed,
  };
}

function totalErrors(e: KernelErr): number {
  return e.compile_errors + e.correct_errors;
}

function resolutionRate(e: KernelErr): number | null {
  const t = totalErrors(e);
  if (t === 0) return null;
  return (100 * e.resolved) / t;
}

function errorShare(e: KernelErr, kind: "compile" | "correct"): number | null {
  const t = totalErrors(e);
  if (t === 0) return null;
  const n = kind === "compile" ? e.compile_errors : e.correct_errors;
  return (100 * n) / t;
}

function typeResolutionRate(e: KernelErr, kind: "compile" | "correct"): number | null {
  const n = kind === "compile" ? e.compile_errors : e.correct_errors;
  if (n === 0) return null;
  const fixed = kind === "compile" ? e.compile_resolved : e.correct_resolved;
  return (100 * fixed) / n;
}

function fmtCountPct(count: number, pct: number | null | undefined): string {
  if (count === 0) return "0";
  const p = fmtPct(pct);
  return p === "—" ? String(count) : `${count} (${p})`;
}

function fmtPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(0) + "%";
}

function fmtNum(v: number): string {
  return Number.isFinite(v) ? String(v) : "—";
}

function emptyTokens(): TokenBucket {
  return { input: 0, output: 0, total: 0, calls: 0 };
}

function fmtTokens(n: number | null | undefined): string {
  if (n == null || typeof n !== "number" || !Number.isFinite(n)) return "—";
  const v = Math.round(n);
  if (Math.abs(v) >= 1_000_000) return (v / 1_000_000).toFixed(2) + "M";
  if (Math.abs(v) >= 1_000) return (v / 1_000).toFixed(1) + "k";
  return String(v);
}

function tokensForIds(col: string, ids: number[]): TokenBucket | null {
  const t = TOKENS[col];
  if (!t) return null;
  const acc = emptyTokens();
  let any = false;
  for (const id of ids) {
    const b = t.byKernel?.[String(id)];
    if (!b) continue;
    any = true;
    acc.input += b.input;
    acc.output += b.output;
    acc.total += b.total;
    acc.calls += b.calls;
  }
  return any ? acc : null;
}

function tokensForKernel(col: string, id: number): TokenBucket | null {
  return TOKENS[col]?.byKernel?.[String(id)] ?? null;
}

function sumTokenBuckets(parts: Array<TokenBucket | null | undefined>): TokenBucket | null {
  const acc = emptyTokens();
  let any = false;
  for (const b of parts) {
    if (!b) continue;
    any = true;
    acc.input += b.input;
    acc.output += b.output;
    acc.total += b.total;
    acc.calls += b.calls;
  }
  return any ? acc : null;
}

function filterExps(
  cohort: Cohort,
  backend: string,
  method: string,
  model: string,
): ExpMeta[] {
  const ids = new Set(cohortIds(cohort));
  return EXPS.filter((e) => {
    if (backend !== "all" && e.backend !== backend) return false;
    if (method !== "all" && e.method !== method) return false;
    if (model !== "all" && e.model !== model) return false;
    if (cohort === "new8") {
      const rows = ERRORS[e.col] ?? {};
      const has = Object.keys(rows).some((k) => ids.has(Number(k)));
      if (!has) return false;
    }
    return true;
  });
}

function aggregateExp(col: string, ids: number[], typeFilter: string): KernelErr {
  const rows = ERRORS[col] ?? {};
  let acc = emptyErr();
  for (const id of ids) {
    if (typeFilter !== "all") {
      const kt = kernelMeta(id)?.ktype ?? "other";
      if (kt !== typeFilter) continue;
    }
    const row = rows[String(id)];
    if (row) acc = mergeErr(acc, row);
  }
  return acc;
}

function kernelRowsForExp(col: string, ids: number[], typeFilter: string): Array<{ id: number; err: KernelErr }> {
  const rows = ERRORS[col] ?? {};
  return ids
    .filter((id) => typeFilter === "all" || (kernelMeta(id)?.ktype ?? "other") === typeFilter)
    .map((id) => ({ id, err: rows[String(id)] ?? emptyErr() }))
    .filter((r) => r.err.nodes > 0 || r.err.unresolved > 0);
}

export default function ErrorReport() {
  const [cohort, setCohort] = useCanvasState<Cohort>("cohort", "sample19");
  const [view, setView] = useCanvasState<ViewMode>("view", "summary");
  const [backendFilter, setBackendFilter] = useCanvasState("backendFilter", "all");
  const [methodFilter, setMethodFilter] = useCanvasState("methodFilter", "all");
  const [modelFilter, setModelFilter] = useCanvasState("modelFilter", "all");
  const [typeFilter, setTypeFilter] = useCanvasState("typeFilter", "all");
  const [setupCol, setSetupCol] = useCanvasState("setupCol", EXPS[0]?.col ?? "");

  const ids = cohortIds(cohort);
  const filteredExps = filterExps(cohort, backendFilter, methodFilter, modelFilter);
  const activeCol = filteredExps.some((e) => e.col === setupCol)
    ? setupCol
    : filteredExps[0]?.col ?? "";

  const summaryRows = filteredExps.map((e) => {
    const t = aggregateExp(e.col, ids, typeFilter);
    const te = totalErrors(t);
    const typeIds =
      typeFilter === "all"
        ? ids
        : ids.filter((id) => (kernelMeta(id)?.ktype ?? "other") === typeFilter);
    const tok = tokensForIds(e.col, typeIds);
    return {
      col: e.col,
      label: e.label,
      ...t,
      total_errors: te,
      resolution: resolutionRate(t),
      tok_in: tok?.input ?? null,
      tok_out: tok?.output ?? null,
      tok_total: tok?.total ?? null,
    };
  }).sort((a, b) => b.total_errors - a.total_errors);

  const grand = summaryRows.reduce((a, r) => mergeErr(a, r), emptyErr());
  const grandTotal = totalErrors(grand);
  const grandTokens = sumTokenBuckets(
    summaryRows.map((r) =>
      r.tok_total == null
        ? null
        : { input: r.tok_in ?? 0, output: r.tok_out ?? 0, total: r.tok_total, calls: 0 },
    ),
  );

  const kernelDetail = kernelRowsForExp(activeCol, ids, typeFilter)
    .sort((a, b) => totalErrors(b.err) - totalErrors(a.err));

  const compareTop = summaryRows.slice(0, 12);

  const setupOptions = filteredExps.map((e) => ({ value: e.col, label: e.label }));

  const kernelTableRows = kernelDetail.map(({ id, err }) => {
    const tok = tokensForKernel(activeCol, id);
    return [
      String(id),
      kernelMeta(id)?.ktype ?? "other",
      err.ever_passed ? "yes" : "no",
      fmtNum(err.nodes),
      fmtNum(err.nodes_pass),
      fmtNum(err.nodes_fail),
      fmtCountPct(err.compile_errors, errorShare(err, "compile")),
      fmtPct(typeResolutionRate(err, "compile")),
      fmtCountPct(err.correct_errors, errorShare(err, "correct")),
      fmtPct(typeResolutionRate(err, "correct")),
      fmtNum(err.codegen_regens),
      fmtNum(err.timeouts),
      fmtNum(err.resolved),
      fmtNum(err.unresolved),
      fmtPct(resolutionRate(err)),
      fmtNum(err.max_depth),
      fmtTokens(tok?.input),
      fmtTokens(tok?.output),
      fmtTokens(tok?.total),
    ];
  });

  const summaryTableRows = summaryRows.map((r) => [
    r.col,
    fmtNum(r.nodes),
    fmtNum(r.nodes_fail),
    fmtCountPct(r.compile_errors, errorShare(r, "compile")),
    fmtPct(typeResolutionRate(r, "compile")),
    fmtCountPct(r.correct_errors, errorShare(r, "correct")),
    fmtPct(typeResolutionRate(r, "correct")),
    fmtNum(r.codegen_regens),
    fmtNum(r.timeouts),
    fmtNum(r.resolved),
    fmtNum(r.unresolved),
    fmtPct(r.resolution),
    fmtTokens(r.tok_in),
    fmtTokens(r.tok_out),
    fmtTokens(r.tok_total),
  ]);

  const summaryTones = summaryRows.map((r) =>
    r.nodes_fail > 0 ? ("warning" as const) : r.total_errors > 20 ? ("info" as const) : ("success" as const),
  );

  return (
    <Stack gap={20} style={{ padding: 24 }}>
      <Stack gap={6}>
        <H1>Cutegen error report</H1>
        <Text tone="secondary">
          Counts from saved node JSON under saved_nodes/. Compile/correctness errors come from fix history on
          saved depth nodes. Failed depth attempts appear as nodes_fail / unresolved even when history is empty
          (e.g. compile timeouts). Updated {GENERATED}.
        </Text>
      </Stack>

      <Callout tone="info" title="How to read this">
        <Text size="sm">
          <Code>Compile</Code> / <Code>Correct</Code> = fix-loop errors recorded in node history.
          <Code>Compile %</Code> / <Code>Correct %</Code> = share of compile+correct errors.
          <Code>Compile fixed</Code> / <Code>Correct fixed</Code> = resolved / that error type.
          <Code>Codegen regen</Code> = codegen retries (history type NONE), not compile/correct fixes.
        </Text>
      </Callout>

      <Row gap={12} wrap>
        <Select
          value={view}
          onChange={(v) => setView(v as ViewMode)}
          options={[
            { value: "summary", label: "View: Setup summary" },
            { value: "kernels", label: "View: Per-kernel detail" },
            { value: "compare", label: "View: Compare setups" },
          ]}
        />
        <Select
          value={cohort}
          onChange={(v) => setCohort(v as Cohort)}
          options={[
            { value: "sample19", label: "Cohort: Sample 19" },
            { value: "new8", label: "Cohort: New 8" },
            { value: "all", label: "Cohort: All 27" },
          ]}
        />
        <Select
          value={backendFilter}
          onChange={setBackendFilter}
          options={[
            { value: "all", label: "Backend: All" },
            ...uniqueSorted(EXPS.map((e) => e.backend)).map((b) => ({ value: b, label: `Backend: ${b}` })),
          ]}
        />
        <Select
          value={methodFilter}
          onChange={setMethodFilter}
          options={[
            { value: "all", label: "Method: All" },
            ...uniqueSorted(EXPS.map((e) => e.method)).map((m) => ({ value: m, label: `Method: ${m}` })),
          ]}
        />
        <Select
          value={modelFilter}
          onChange={setModelFilter}
          options={[
            { value: "all", label: "Model: All" },
            { value: "Kimi K3", label: "Model: Kimi K3" },
            { value: "Sonnet 5", label: "Model: Sonnet 5" },
          ]}
        />
        <Select
          value={typeFilter}
          onChange={setTypeFilter}
          options={[
            { value: "all", label: "Kernel type: All" },
            ...KERNEL_TYPES.map((t) => ({ value: t, label: `Kernel type: ${t}` })),
          ]}
        />
        {view === "kernels" && setupOptions.length > 0 ? (
          <Select value={activeCol} onChange={setSetupCol} options={setupOptions} />
        ) : null}
      </Row>

      <Grid columns={6} gap={12}>
        <Stat value={fmtNum(grandTotal)} label="Total compile+correct errors" tone="info" />
        <Stat value={fmtCountPct(grand.compile_errors, errorShare(grand, "compile"))} label="Compile errors (share)" />
        <Stat value={fmtPct(typeResolutionRate(grand, "compile"))} label="Compile fixed %" tone="success" />
        <Stat value={fmtCountPct(grand.correct_errors, errorShare(grand, "correct"))} label="Correct errors (share)" />
        <Stat value={fmtPct(typeResolutionRate(grand, "correct"))} label="Correct fixed %" tone="success" />
        <Stat value={fmtPct(resolutionRate(grand))} label="Overall resolution" />
      </Grid>
      <Grid columns={3} gap={12}>
        <Stat value={fmtTokens(grandTokens?.input)} label="Input tokens (filtered setups)" />
        <Stat value={fmtTokens(grandTokens?.output)} label="Output tokens (filtered setups)" />
        <Stat value={fmtTokens(grandTokens?.total)} label="Total tokens (filtered setups)" tone="info" />
      </Grid>

      {view === "compare" && compareTop.length > 0 ? (
        <Stack gap={8}>
          <H2>Errors by setup (top 12)</H2>
          <Text size="small" tone="secondary">
            Source: saved_nodes · grouped bars = compile vs correctness vs resolved counts per setup
          </Text>
          <BarChart
            categories={compareTop.map((r) => r.col)}
            series={[
              { name: "Compile", data: compareTop.map((r) => r.compile_errors), tone: "danger" },
              { name: "Correctness", data: compareTop.map((r) => r.correct_errors), tone: "warning" },
              { name: "Resolved", data: compareTop.map((r) => r.resolved), tone: "success" },
            ]}
            height={320}
          />
        </Stack>
      ) : null}

      {view === "summary" ? (
        <Stack gap={8}>
          <H2>Setup summary</H2>
          <Text size="small" tone="secondary">
            Cohort {cohort} · {filteredExps.length} setups · counts show n (% of compile+correct); fixed % = resolved for that type
          </Text>
          <Table
            headers={[
              "Setup",
              "Nodes",
              "Fail depths",
              "Compile",
              "Compile fixed",
              "Correct",
              "Correct fixed",
              "Codegen regen",
              "Timeouts",
              "Resolved",
              "Unresolved",
              "Overall",
              "In tok",
              "Out tok",
              "Total tok",
            ]}
            rows={summaryTableRows}
            rowTone={summaryTones}
          />
        </Stack>
      ) : null}

      {view === "kernels" && activeCol ? (
        <Stack gap={8}>
          <H2>Per-kernel errors · {filteredExps.find((e) => e.col === activeCol)?.label ?? activeCol}</H2>
          <Text size="small" tone="secondary">
            {kernelDetail.length} kernels with saved nodes in cohort
          </Text>
          <Table
            headers={[
              "ID",
              "Type",
              "Ever passed",
              "Nodes",
              "Pass",
              "Fail",
              "Compile",
              "Compile fixed",
              "Correct",
              "Correct fixed",
              "Regen",
              "Timeout",
              "Resolved",
              "Unresolved",
              "Overall",
              "Max depth",
              "In tok",
              "Out tok",
              "Total tok",
            ]}
            rows={kernelTableRows}
            rowTone={kernelDetail.map(({ err }) =>
              !err.ever_passed ? ("danger" as const) : err.nodes_fail > 0 ? ("warning" as const) : ("success" as const),
            )}
          />
        </Stack>
      ) : null}
    </Stack>
  );
}
'''


def main() -> None:
    payload_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cutegen_error_payload.json")
    if not payload_path.exists():
        subprocess.run([sys.executable, str(SCRIPTS / "generate_error_payload.py")], check=True)
    payload = json.loads(payload_path.read_text())

    header = f'''import {{
  BarChart,
  Callout,
  Code,
  Grid,
  H1,
  H2,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
}} from "cursor/canvas";

type KernelErr = {{
  nodes: number;
  nodes_pass: number;
  nodes_fail: number;
  compile_errors: number;
  correct_errors: number;
  compile_resolved: number;
  correct_resolved: number;
  codegen_regens: number;
  timeouts: number;
  resolved: number;
  unresolved: number;
  max_depth: number;
  ever_passed: boolean;
}};

type TokenBucket = {{ input: number; output: number; total: number; calls: number }};
type TokenAgg = TokenBucket & {{
  byKernel: Record<string, TokenBucket>;
  sources: string[];
}};

const GENERATED = {json.dumps(payload["generated"])};
const SAMPLE19_IDS = {js(payload["sample19Ids"])};
const NEW8_IDS = {js(payload["new8Ids"])};
const KERNEL_TYPES: string[] = {js(payload["kernelTypes"])};
const EXPS = {js(payload["exps"])};
const KERNELS = {js(payload["kernels"])};
const ERRORS: Record<string, Record<string, KernelErr>> = {js(payload["errors"])};
const TOKENS: Record<string, TokenAgg> = {js(payload.get("tokens") or {})};
'''

    tsx = header + ERROR_REPORT_BODY
    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    out = CANVAS_DIR / "error-report.canvas.tsx"
    out.write_text(tsx)
    print(f"wrote {out} ({len(tsx)} chars)")


if __name__ == "__main__":
    main()
