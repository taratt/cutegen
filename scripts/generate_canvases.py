#!/usr/bin/env python3
"""Generate Cutegen canvas TSX files from canvas payload JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CANVAS_DIR = Path.home() / ".cursor/projects/home-tarasaba-PycharmProjects-cutegen-test/canvases"


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
        if all(isinstance(x, (int, float)) or x is None for x in obj):
            inner = ", ".join("null" if x is None else repr(x) for x in obj)
            return f"[{inner}]"
        items = ",\n".join(sp + "  " + js(x, indent + 1) for x in obj)
        return f"[\n{items}\n{sp}]"
    if isinstance(obj, dict):
        items = ",\n".join(f'{sp}  {json.dumps(k)}: {js(v, indent + 1)}' for k, v in obj.items())
        return f"{{\n{items}\n{sp}}}"
    return json.dumps(obj)


SHARED_HELPERS = '''
const ABS_MAX_DEPTH = MAX_DEPTH;

function sliceCurve(pts: (number | null)[], maxDepth: number): (number | null)[] {
  return pts.slice(0, maxDepth + 1);
}

function depthLabels(maxDepth: number): string[] {
  return DEPTH_LABELS.slice(0, maxDepth + 1);
}

function validInRange(pts: (number | null)[] | undefined, maxDepth: number): number[] {
  if (!pts) return [];
  return sliceCurve(pts, maxDepth).filter((p): p is number => p != null && Number.isFinite(p));
}

function kernelAtDepth(pts: (number | null)[] | undefined, maxDepth: number) {
  const speeds = validInRange(pts, maxDepth);
  const best = speeds.length ? Math.max(...speeds) : null;
  const correct = speeds.length > 0;
  const fast1 = correct && best != null && best > 1;
  return { best, correct, fast1, nDepths: speeds.length };
}

function isPrecisionCheat(col: string, kernelId: number): boolean {
  return PRECISION_CHEATS.some((c) => c.col === col && c.kernel === kernelId);
}

function kernelAtDepthFiltered(
  pts: (number | null)[] | undefined,
  maxDepth: number,
  col: string,
  kernelId: number,
  excludePrecisionCheats: boolean,
) {
  const base = kernelAtDepth(pts, maxDepth);
  if (!excludePrecisionCheats || !isPrecisionCheat(col, kernelId)) return base;
  return { best: null, correct: false, fast1: false, nDepths: base.nDepths };
}

function bestSpeedupInRange(pts: (number | null)[] | undefined, maxDepth: number): number | null {
  return kernelAtDepth(pts, maxDepth).best;
}

function initialSpeedupInRange(pts: (number | null)[] | undefined, maxDepth: number): number | null {
  if (!pts || maxDepth < 0) return null;
  const p = pts[0];
  return p != null && Number.isFinite(p) && p > 0 ? p : null;
}

function maxOverInitialRatio(pts: (number | null)[] | undefined, maxDepth: number): number | null {
  const initial = initialSpeedupInRange(pts, maxDepth);
  const best = bestSpeedupInRange(pts, maxDepth);
  if (initial == null || best == null || initial <= 0) return null;
  return best / initial;
}

function fmtGain(v: number | null | undefined): string {
  if (v == null || typeof v !== "number" || !Number.isFinite(v) || v <= 0) return "—";
  return v.toFixed(2) + "×";
}

function finalSpeedupInRange(pts: (number | null)[] | undefined, maxDepth: number): number | null {
  if (!pts) return null;
  for (let i = Math.min(maxDepth, pts.length - 1); i >= 0; i--) {
    const p = pts[i];
    if (p != null && Number.isFinite(p)) return p;
  }
  return null;
}

function depthAtBestInRange(pts: (number | null)[] | undefined, maxDepth: number): number | null {
  if (!pts) return null;
  let best: number | null = null;
  let bestD: number | null = null;
  for (let d = 0; d <= maxDepth && d < pts.length; d++) {
    const p = pts[d];
    if (p != null && Number.isFinite(p) && (best == null || p > best)) {
      best = p;
      bestD = d;
    }
  }
  return bestD;
}

function toSeriesData(pts: (number | null)[], maxDepth: number): number[] {
  return sliceCurve(pts, maxDepth).map((p) => (p == null ? 0 : p));
}

function hasAnyDataInRange(pts: (number | null)[] | undefined, maxDepth: number): boolean {
  return validInRange(pts, maxDepth).length > 0;
}

function maxDepthOptions(): { value: string; label: string }[] {
  return Array.from({ length: ABS_MAX_DEPTH + 1 }, (_, d) => ({
    value: String(d),
    label: d === ABS_MAX_DEPTH ? `Depth 0–${d} (all)` : `Depth 0–${d}`,
  }));
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

/** Sum tokens for a setup over the given kernel IDs (cohort filter). */
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
'''

TOKEN_TYPES = '''
type TokenBucket = { input: number; output: number; total: number; calls: number };
type TokenAgg = TokenBucket & {
  byKernel: Record<string, TokenBucket>;
  sources: string[];
};
'''

SPEEDUP_DEPTH_BODY = '''
type Cohort = "sample19" | "new8" | "all";
type ViewMode = "setup" | "kernel" | "grid" | "backend-grid";

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

function kernelName(id: number): string {
  return kernelMeta(id)?.name ?? `${id}_`;
}

function kernelType(id: number): string {
  return kernelMeta(id)?.ktype ?? "other";
}

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values)].sort();
}

function filterIdsByCategory(ids: number[], category: string): number[] {
  if (category === "all") return ids;
  return ids.filter((id) => kernelType(id) === category);
}

function setupShortLabel(e: ExpMeta): string {
  const methodShort =
    e.method === "Minimal"
      ? "min"
      : e.method === "Minimal from start"
        ? "minstart"
        : e.method === "No profiling"
      ? "nopf"
      : e.method === "Delayed profiling"
        ? "delay"
        : e.method === "Profiling from start"
          ? "start"
          : e.method;
  const modelShort = e.model === "Sonnet 5" ? "Sonnet" : e.model === "Kimi K3" ? "Kimi" : e.model === "GPT-5" ? "GPT-5" : e.model;
  return `${e.backend} · ${methodShort} · ${modelShort}`;
}

function filterSetups(
  exps: ExpMeta[],
  backend: string,
  method: string,
  model: string,
): ExpMeta[] {
  return exps.filter((e) => {
    if (backend !== "all" && e.backend !== backend) return false;
    if (method !== "all" && e.method !== method) return false;
    if (model !== "all" && e.model !== model) return false;
    return true;
  });
}

export default function SpeedupVsDepthCanvas() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<ViewMode>("view", "kernel");
  const [cohort, setCohort] = useCanvasState<Cohort>("cohort", "sample19");
  const [expCol, setExpCol] = useCanvasState<string>("exp", EXPS[0]?.col ?? "");
  const [expColB, setExpColB] = useCanvasState<string>("expB", EXPS[1]?.col ?? "none");
  const [expColC, setExpColC] = useCanvasState<string>("expC", "none");
  const [kernelId, setKernelId] = useCanvasState<number>("kernel", SAMPLE19_IDS[0]);
  const [maxDepth, setMaxDepth] = useCanvasState<number>("maxDepth", ABS_MAX_DEPTH);
  const [backendFilter, setBackendFilter] = useCanvasState<string>("backendFilter", "CuTe");
  const [methodFilter, setMethodFilter] = useCanvasState<string>("methodFilter", "all");
  const [modelFilter, setModelFilter] = useCanvasState<string>("modelFilter", "all");
  const [categoryFilter, setCategoryFilter] = useCanvasState<string>("categoryFilter", "all");

  const cohortPool = cohortIds(cohort);
  const categories = uniqueSorted([
    ...KERNEL_TYPES,
    ...KERNELS.map((k) => k.ktype),
  ]);
  const ids = filterIdsByCategory(cohortPool, categoryFilter);
  const activeKernelId = ids.includes(kernelId) ? kernelId : (ids[0] ?? kernelId);
  const backends = uniqueSorted(EXPS.map((e) => e.backend));
  const methods = uniqueSorted(EXPS.map((e) => e.method));
  const models = uniqueSorted(EXPS.map((e) => e.model));

  const filteredExps = filterSetups(EXPS, backendFilter, methodFilter, modelFilter);
  const exp = EXPS.find((e) => e.col === expCol) ?? EXPS[0];
  const expCurves = CURVES[exp?.col ?? ""] ?? {};
  const cats = depthLabels(maxDepth);

  const gridSetupCols = [expCol, expColB, expColC].filter(
    (c, i, arr) => c && c !== "none" && arr.indexOf(c) === i,
  );
  const gridSetups = gridSetupCols
    .map((col) => EXPS.find((e) => e.col === col))
    .filter((e): e is ExpMeta => e != null);

  const setupOptions = [
    ...EXPS.map((e) => ({ value: e.col, label: e.label })),
  ];
  const optionalSetupOptions = [
    { value: "none", label: "— none —" },
    ...setupOptions,
  ];

  const setupSeries = ids
    .filter((id) => hasAnyDataInRange(expCurves[String(id)], maxDepth))
    .map((id) => ({
      name: kernelLabel(id),
      data: toSeriesData(expCurves[String(id)] ?? DEPTH_LABELS.map(() => null), maxDepth),
    }));

  const matchedKernelExps = filteredExps.filter((e) =>
    hasAnyDataInRange(CURVES[e.col]?.[String(activeKernelId)], maxDepth),
  );

  const kernelSeries = matchedKernelExps.map((e) => ({
    name: setupShortLabel(e),
    data: toSeriesData(CURVES[e.col][String(activeKernelId)], maxDepth),
  }));

  const kernelTableRows = matchedKernelExps
    .map((e) => {
      const pts = CURVES[e.col][String(activeKernelId)];
      const tok = tokensForKernel(e.col, activeKernelId);
      return {
        setup: setupShortLabel(e),
        backend: e.backend,
        method: e.method,
        model: e.model,
        best: bestSpeedupInRange(pts, maxDepth),
        initial: initialSpeedupInRange(pts, maxDepth),
        gain: maxOverInitialRatio(pts, maxDepth),
        final: finalSpeedupInRange(pts, maxDepth),
        depthBest: depthAtBestInRange(pts, maxDepth),
        depths: validInRange(pts, maxDepth).length,
        tokIn: tok?.input ?? null,
        tokOut: tok?.output ?? null,
        tokTotal: tok?.total ?? null,
        path: e.path,
      };
    })
    .sort((a, b) => (b.best ?? 0) - (a.best ?? 0));

  const setupTokens = tokensForIds(exp?.col ?? "", ids);
  const kernelTokensTotal = sumTokenBuckets(kernelTableRows.map((r) =>
    r.tokTotal == null
      ? null
      : { input: r.tokIn ?? 0, output: r.tokOut ?? 0, total: r.tokTotal, calls: 0 },
  ));

  const gridKernels = ids.filter((id) =>
    gridSetups.some((e) => hasAnyDataInRange(CURVES[e.col]?.[String(id)], maxDepth)),
  );

  const backendGridExps = filterSetups(
    EXPS,
    backendFilter === "all" ? "CuTe" : backendFilter,
    methodFilter,
    modelFilter,
  ).filter((e) => hasAnyDataInRange(CURVES[e.col]?.[String(activeKernelId)], maxDepth));

  const setupStats = {
    n: setupSeries.length,
    best: setupSeries.reduce<number | null>((acc, s) => {
      const vals = s.data.filter((x) => x > 0);
      if (!vals.length) return acc;
      const b = Math.max(...vals);
      return acc == null ? b : Math.max(acc, b);
    }, null),
    fast1: setupSeries.filter((s) => Math.max(...s.data.filter((x) => x > 0)) > 1).length,
  };

  const kernelStats = {
    n: kernelSeries.length,
    best: kernelTableRows[0]?.best ?? null,
    bestSetup: kernelTableRows[0]?.setup ?? "—",
    fast1: kernelTableRows.filter((r) => (r.best ?? 0) > 1).length,
  };

  const tableRows = ids
    .filter((id) =>
      view === "grid"
        ? gridSetups.some((e) => hasAnyDataInRange(CURVES[e.col]?.[String(id)], maxDepth))
        : hasAnyDataInRange(expCurves[String(id)], maxDepth),
    )
    .map((id) => {
      if (view === "grid" && gridSetups.length > 1) {
        const bests = Object.fromEntries(
          gridSetups.map((e) => [
            e.col,
            bestSpeedupInRange(CURVES[e.col]?.[String(id)], maxDepth),
          ]),
        );
        const gains = Object.fromEntries(
          gridSetups.map((e) => [
            e.col,
            maxOverInitialRatio(CURVES[e.col]?.[String(id)], maxDepth),
          ]),
        );
        const vals = Object.values(bests).filter((v): v is number => v != null);
        const gainVals = Object.values(gains).filter((v): v is number => v != null);
        return {
          id,
          name: kernelName(id),
          best: vals.length ? Math.max(...vals) : null,
          initial: null as number | null,
          gain: gainVals.length ? Math.max(...gainVals) : null,
          final: null as number | null,
          depthBest: null as number | null,
          depths: gridSetups.reduce(
            (n, e) => n + validInRange(CURVES[e.col]?.[String(id)], maxDepth).length,
            0,
          ),
          bests,
          gains,
        };
      }
      const pts = expCurves[String(id)];
      const tok = tokensForKernel(exp?.col ?? "", id);
      return {
        id,
        name: kernelName(id),
        best: bestSpeedupInRange(pts, maxDepth),
        initial: initialSpeedupInRange(pts, maxDepth),
        gain: maxOverInitialRatio(pts, maxDepth),
        final: finalSpeedupInRange(pts, maxDepth),
        depthBest: depthAtBestInRange(pts, maxDepth),
        depths: validInRange(pts, maxDepth).length,
        tokIn: tok?.input ?? null,
        tokOut: tok?.output ?? null,
        tokTotal: tok?.total ?? null,
        bests: {} as Record<string, number | null>,
        gains: {} as Record<string, number | null>,
      };
    })
    .sort((a, b) => (b.best ?? 0) - (a.best ?? 0));

  const gridCompareLabel = gridSetups.map((e) => setupShortLabel(e)).join(" vs ");

  const scopeLabel = [
    categoryFilter === "all" ? null : categoryFilter,
    backendFilter === "all" ? "all backends" : backendFilter,
    methodFilter === "all" ? null : methodFilter,
    modelFilter === "all" ? null : modelFilter,
  ]
    .filter(Boolean)
    .join(" · ");

  const categoryScope =
    categoryFilter === "all" ? "all categories" : categoryFilter;

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>Speedup vs search depth</H1>
        <Text tone="secondary">
          Correct compile+timing nodes only · speedup = ref_time / gen_time · considering depths 0–{maxDepth} ·
          generated {GENERATED}
        </Text>
      </Stack>

      <Row gap={12} wrap>
        <Select
          label="Max depth"
          value={String(maxDepth)}
          onChange={(v) => setMaxDepth(Number(v))}
          options={maxDepthOptions()}
        />
        <Select
          label="View"
          value={view}
          onChange={(v) => setView(v as ViewMode)}
          options={[
            { value: "kernel", label: "One kernel — compare setups" },
            { value: "backend-grid", label: "One kernel — setup cards" },
            { value: "setup", label: "One setup — all kernels" },
            { value: "grid", label: "Kernel grid — 1–3 setups" },
          ]}
        />
        <Select
          label="Cohort"
          value={cohort}
          onChange={(v) => setCohort(v as Cohort)}
          options={[
            { value: "sample19", label: "Sample 19" },
            { value: "new8", label: "New 8" },
            { value: "all", label: "All 27" },
          ]}
        />
        <Select
          label="Category"
          value={categoryFilter}
          onChange={(v) => {
            setCategoryFilter(v);
            const nextIds = filterIdsByCategory(cohortPool, v);
            if (nextIds.length && !nextIds.includes(kernelId)) {
              setKernelId(nextIds[0]);
            }
          }}
          options={[
            { value: "all", label: "All categories" },
            ...categories.map((c) => ({
              value: c,
              label: `${c} (${filterIdsByCategory(cohortPool, c).length})`,
            })),
          ]}
        />
        {(view === "kernel" || view === "backend-grid") && (
          <>
            <Select
              label="Kernel"
              value={String(activeKernelId)}
              onChange={(v) => setKernelId(Number(v))}
              options={ids.map((id) => ({ value: String(id), label: `${kernelLabel(id)} · ${kernelName(id)}` }))}
            />
            <Select
              label="Backend"
              value={backendFilter}
              onChange={setBackendFilter}
              options={[
                { value: "all", label: "All backends" },
                ...backends.map((b) => ({ value: b, label: `${b} only` })),
              ]}
            />
            <Select
              label="Method"
              value={methodFilter}
              onChange={setMethodFilter}
              options={[
                { value: "all", label: "All methods" },
                ...methods.map((m) => ({ value: m, label: m })),
              ]}
            />
            <Select
              label="Model"
              value={modelFilter}
              onChange={setModelFilter}
              options={[
                { value: "all", label: "All models" },
                ...models.map((m) => ({ value: m, label: m })),
              ]}
            />
          </>
        )}
        {view === "setup" && (
          <Select
            label="Setup"
            value={expCol}
            onChange={setExpCol}
            options={setupOptions}
          />
        )}
        {view === "grid" && (
          <>
            <Select
              label="Setup A"
              value={expCol}
              onChange={setExpCol}
              options={setupOptions}
            />
            <Select
              label="Setup B"
              value={expColB}
              onChange={setExpColB}
              options={optionalSetupOptions}
            />
            <Select
              label="Setup C"
              value={expColC}
              onChange={setExpColC}
              options={optionalSetupOptions}
            />
          </>
        )}
      </Row>

      {view === "setup" && setupSeries.length > 0 && (
        <Stack gap={8}>
          <H2>
            {exp?.label ?? expCol} · {categoryScope}
          </H2>
          <Row gap={16} wrap>
            <Stat label="Kernels plotted" value={String(setupStats.n)} />
            <Stat
              label={`Best speedup (0–${maxDepth})`}
              value={setupStats.best?.toFixed(2) ?? "—"}
              suffix="×"
              tone="success"
            />
            <Stat label="Fast (>1×)" value={`${setupStats.fast1}/${setupStats.n}`} />
            <Stat label="Input tokens" value={fmtTokens(setupTokens?.input)} />
            <Stat label="Output tokens" value={fmtTokens(setupTokens?.output)} />
            <Stat label="Total tokens" value={fmtTokens(setupTokens?.total)} tone="info" />
          </Row>
          <LineChart
            categories={cats}
            series={setupSeries}
            height={420}
            valueSuffix="×"
            style={{ border: `1px solid ${theme.stroke}`, borderRadius: 8, padding: 12 }}
          />
          <Text tone="secondary" size="sm">
            X-axis: optimization depth. Y-axis: speedup vs PyTorch reference. Stats and chart limited to depths 0–{maxDepth}.
            Category filter: {categoryScope}. Token totals are summed over plotted cohort kernels for this setup.
          </Text>
        </Stack>
      )}

      {view === "kernel" && kernelSeries.length > 0 && (
        <Stack gap={8}>
          <H2>
            {kernelLabel(activeKernelId)} — {kernelName(activeKernelId)}
          </H2>
          <Text tone="secondary">Comparing {scopeLabel}</Text>
          <Row gap={16} wrap>
            <Stat label="Setups plotted" value={String(kernelStats.n)} />
            <Stat
              label={`Best among setups (0–${maxDepth})`}
              value={kernelStats.best?.toFixed(2) ?? "—"}
              suffix="×"
              tone="success"
            />
            <Stat label="Best setup" value={kernelStats.bestSetup} />
            <Stat label="Fast (>1×)" value={`${kernelStats.fast1}/${kernelStats.n}`} />
            <Stat label="Input tokens (Σ setups)" value={fmtTokens(kernelTokensTotal?.input)} />
            <Stat label="Output tokens (Σ setups)" value={fmtTokens(kernelTokensTotal?.output)} />
            <Stat label="Total tokens (Σ setups)" value={fmtTokens(kernelTokensTotal?.total)} tone="info" />
          </Row>
          <LineChart
            categories={cats}
            series={kernelSeries}
            height={420}
            valueSuffix="×"
            style={{ border: `1px solid ${theme.stroke}`, borderRadius: 8, padding: 12 }}
          />
          <Text tone="secondary" size="sm">
            Each series is one experiment setup · depths 0–{maxDepth}. Use Category/Backend/Method/Model to narrow.
          </Text>
          <Divider />
          <H2>Setup ranking for this kernel</H2>
          <Table
            columns={[
              { key: "setup", header: "Setup" },
              { key: "method", header: "Method" },
              { key: "model", header: "Model" },
              { key: "best", header: "Best ×", align: "right" },
              { key: "initial", header: "Init ×", align: "right" },
              { key: "gain", header: "Max/init", align: "right" },
              { key: "depthBest", header: "At depth", align: "right" },
              { key: "final", header: "Final ×", align: "right" },
              { key: "depths", header: "# depths", align: "right" },
              { key: "tokIn", header: "In tok", align: "right" },
              { key: "tokOut", header: "Out tok", align: "right" },
              { key: "tokTotal", header: "Total tok", align: "right" },
            ]}
            rows={kernelTableRows.map((r) => ({
              ...r,
              best: r.best?.toFixed(3) ?? "—",
              initial: r.initial?.toFixed(3) ?? "—",
              gain: fmtGain(r.gain),
              final: r.final?.toFixed(3) ?? "—",
              depthBest: r.depthBest ?? "—",
              tokIn: fmtTokens(r.tokIn),
              tokOut: fmtTokens(r.tokOut),
              tokTotal: fmtTokens(r.tokTotal),
            }))}
          />
        </Stack>
      )}

      {view === "backend-grid" && backendGridExps.length > 0 && (
        <Stack gap={8}>
          <H2>
            {kernelLabel(activeKernelId)} — {scopeLabel || (backendFilter === "all" ? "CuTe" : backendFilter)} setups
          </H2>
          <Grid columns={2} gap={12}>
            {backendGridExps.map((e) => {
              const pts = CURVES[e.col][String(activeKernelId)] ?? [];
              const best = bestSpeedupInRange(pts, maxDepth);
              return (
                <Stack
                  key={e.col}
                  gap={4}
                  style={{ border: `1px solid ${theme.stroke}`, borderRadius: 8, padding: 10 }}
                >
                  <Text weight="semibold">{setupShortLabel(e)}</Text>
                  <Text tone="secondary" size="sm">
                    {e.method} · {e.model} · best {best?.toFixed(2) ?? "—"}× @ depth{" "}
                    {depthAtBestInRange(pts, maxDepth) ?? "—"}
                  </Text>
                  <LineChart
                    categories={cats}
                    series={[{ name: "speedup", data: toSeriesData(pts, maxDepth) }]}
                    height={160}
                    valueSuffix="×"
                  />
                </Stack>
              );
            })}
          </Grid>
        </Stack>
      )}

      {view === "grid" && gridKernels.length > 0 && gridSetups.length > 0 && (
        <Stack gap={8}>
          <H2>
            {gridCompareLabel} · {categoryScope} — per-kernel trajectories (0–{maxDepth})
          </H2>
          <Text tone="secondary" size="sm">
            Overlaying {gridSetups.length} setup{gridSetups.length === 1 ? "" : "s"} on each kernel card.
            Leave Setup B/C as none for a single-setup grid.
          </Text>
          <Grid columns={3} gap={12}>
            {gridKernels.map((id) => {
              const series = gridSetups
                .filter((e) => hasAnyDataInRange(CURVES[e.col]?.[String(id)], maxDepth))
                .map((e) => ({
                  name: setupShortLabel(e),
                  data: toSeriesData(CURVES[e.col][String(id)], maxDepth),
                }));
              const bests = gridSetups.map((e) => ({
                label: setupShortLabel(e),
                best: bestSpeedupInRange(CURVES[e.col]?.[String(id)], maxDepth),
              }));
              const bestLine = bests
                .map((b) => `${b.label} ${b.best?.toFixed(2) ?? "—"}×`)
                .join(" · ");
              return (
                <Stack
                  key={id}
                  gap={4}
                  style={{ border: `1px solid ${theme.stroke}`, borderRadius: 8, padding: 10 }}
                >
                  <Text weight="semibold">{kernelLabel(id)}</Text>
                  <Text tone="secondary" size="sm">
                    {bestLine}
                  </Text>
                  <LineChart
                    categories={cats}
                    series={series}
                    height={160}
                    valueSuffix="×"
                  />
                </Stack>
              );
            })}
          </Grid>
        </Stack>
      )}

      {(view === "setup" || view === "grid") && tableRows.length > 0 && (
        <Stack gap={8}>
          <Divider />
          <H2>
            Summary — {view === "grid" ? gridCompareLabel : (exp?.label ?? expCol)} · {categoryScope}{" "}
            (depths 0–{maxDepth})
          </H2>
          <Table
            columns={
              view === "grid" && gridSetups.length > 1
                ? [
                    { key: "id", header: "ID", width: 48 },
                    { key: "name", header: "Kernel" },
                    ...gridSetups.flatMap((e) => [
                      {
                        key: e.col,
                        header: `${setupShortLabel(e)} ×`,
                        align: "right" as const,
                      },
                      {
                        key: `${e.col}_gain`,
                        header: `${setupShortLabel(e)} max/init`,
                        align: "right" as const,
                      },
                    ]),
                    { key: "best", header: "Best of · ×", align: "right" as const },
                    { key: "gain", header: "Best max/init", align: "right" as const },
                  ]
                : [
                    { key: "id", header: "ID", width: 48 },
                    { key: "name", header: "Kernel" },
                    { key: "best", header: "Best ×", align: "right" as const },
                    { key: "initial", header: "Init ×", align: "right" as const },
                    { key: "gain", header: "Max/init", align: "right" as const },
                    { key: "depthBest", header: "At depth", align: "right" as const },
                    { key: "final", header: "Final ×", align: "right" as const },
                    { key: "depths", header: "# depths", align: "right" as const },
                    { key: "tokIn", header: "In tok", align: "right" as const },
                    { key: "tokOut", header: "Out tok", align: "right" as const },
                    { key: "tokTotal", header: "Total tok", align: "right" as const },
                  ]
            }
            rows={tableRows.map((r) => {
              if (view === "grid" && gridSetups.length > 1) {
                const row: Record<string, string | number> = {
                  id: r.id,
                  name: r.name,
                  best: r.best?.toFixed(3) ?? "—",
                  gain: fmtGain(r.gain),
                };
                for (const e of gridSetups) {
                  row[e.col] = r.bests[e.col]?.toFixed(3) ?? "—";
                  row[`${e.col}_gain`] = fmtGain(r.gains[e.col]);
                }
                return row;
              }
              return {
                ...r,
                best: r.best?.toFixed(3) ?? "—",
                initial: r.initial?.toFixed(3) ?? "—",
                gain: fmtGain(r.gain),
                final: r.final?.toFixed(3) ?? "—",
                depthBest: r.depthBest ?? "—",
                tokIn: fmtTokens(r.tokIn),
                tokOut: fmtTokens(r.tokOut),
                tokTotal: fmtTokens(r.tokTotal),
              };
            })}
          />
        </Stack>
      )}

      {((view === "setup" && setupSeries.length === 0) ||
        (view === "kernel" && kernelSeries.length === 0) ||
        (view === "backend-grid" && backendGridExps.length === 0) ||
        (view === "grid" && (gridKernels.length === 0 || gridSetups.length === 0))) && (
        <Callout tone="warning">
          No curve data for this selection at depths 0–{maxDepth}. Try other setups or widen Category, or the run may still be in progress.
        </Callout>
      )}
    </Stack>
  );
}
'''

SCOREBOARD_BODY = '''
type ExpMeta = {
  col: string;
  path: string;
  label: string;
  backend: string;
  method: string;
  model: string;
  leaf: string;
};

type KernelMeta = { id: number; name: string; ktype: string; cohort: string };

type CohortStats = {
  target_n: number;
  attempted_n: number;
  correct_n: number;
  correct_pct: number;
  fast1_n: number;
  fast1_pct: number;
  mean_speedup: number | null;
  median_speedup: number | null;
  best_speedup: number | null;
  mean_gain: number | null;
  tok_in: number | null;
  tok_out: number | null;
  tok_total: number | null;
};

function fmtSp(v: number | null | undefined): string {
  if (v == null || typeof v !== "number" || Number.isNaN(v)) return "—";
  return v.toFixed(2) + "×";
}

function fmtPct(v: number): string {
  return v.toFixed(1) + "%";
}

type Cohort = "sample19" | "new8" | "all";

function cohortIds(cohort: Cohort): number[] {
  if (cohort === "sample19") return SAMPLE19_IDS;
  if (cohort === "new8") return NEW8_IDS;
  return [...SAMPLE19_IDS, ...NEW8_IDS];
}

function cohortLabel(cohort: Cohort): string {
  if (cohort === "sample19") return "sample-19";
  if (cohort === "new8") return "new-8";
  return "all-27";
}

function summarizeExp(
  exp: ExpMeta,
  cohort: Cohort,
  typeFilter: string,
  maxDepth: number,
  excludePrecisionCheats: boolean,
): ExpMeta & CohortStats {
  const ids = cohortIds(cohort);
  const typeIds =
    typeFilter === "all" ? ids : ids.filter((id) => KERNELS.find((k) => k.id === id)?.ktype === typeFilter);
  const target_n = typeIds.length;
  const curves = CURVES[exp.col] ?? {};
  const pool = typeIds.filter((id) => curves[String(id)] != null);
  const stats = pool.map((id) => kernelAtDepthFiltered(curves[String(id)], maxDepth, exp.col, id, excludePrecisionCheats));
  const correct_n = stats.filter((s) => s.correct).length;
  const fast1_n = stats.filter((s) => s.fast1).length;
  const speeds = stats.filter((s) => s.correct && s.best != null).map((s) => s.best as number);
  const gains = pool
    .filter((id) => !excludePrecisionCheats || !isPrecisionCheat(exp.col, id))
    .map((id) => maxOverInitialRatio(curves[String(id)], maxDepth))
    .filter((g): g is number => g != null);
  const mean = speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : null;
  const sorted = speeds.slice().sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
  const tok = tokensForIds(exp.col, typeIds);
  return {
    ...exp,
    target_n,
    attempted_n: pool.length,
    correct_n,
    correct_pct: target_n ? (100 * correct_n) / target_n : 0,
    fast1_n,
    fast1_pct: target_n ? (100 * fast1_n) / target_n : 0,
    mean_speedup: mean,
    median_speedup: median,
    best_speedup: speeds.length ? Math.max(...speeds) : null,
    mean_gain: gains.length ? gains.reduce((a, b) => a + b, 0) / gains.length : null,
    tok_in: tok?.input ?? null,
    tok_out: tok?.output ?? null,
    tok_total: tok?.total ?? null,
  };
}

function buildMatrix(cohort: Cohort, typeFilter: string, maxDepth: number, cols: string[], excludePrecisionCheats: boolean) {
  const ids = cohortIds(cohort);
  const filteredIds =
    typeFilter === "all" ? ids : ids.filter((id) => KERNELS.find((k) => k.id === id)?.ktype === typeFilter);
  return filteredIds.map((id) => {
    const k = KERNELS.find((x) => x.id === id);
    const row: Record<string, string | number | boolean | null> = {
      id,
      name: k?.name ?? `${id}_`,
      ktype: k?.ktype ?? "other",
    };
    for (const col of cols) {
      const pts = CURVES[col]?.[String(id)];
      const st = kernelAtDepthFiltered(pts, maxDepth, col, id, excludePrecisionCheats);
      row[col] = st.best;
      row[`${col}_gain`] = excludePrecisionCheats && isPrecisionCheat(col, id)
        ? null
        : maxOverInitialRatio(pts, maxDepth);
      row[`${col}_correct`] = st.correct;
      row[`${col}_fast1`] = st.fast1;
      row[`${col}_cheat`] = isPrecisionCheat(col, id);
    }
    const gainVals = cols
      .map((col) => row[`${col}_gain`])
      .filter((g): g is number => typeof g === "number");
    row.best_gain = gainVals.length ? Math.max(...gainVals) : null;
    return row;
  });
}

function cellSp(row: Record<string, unknown>, col: string): string {
  const v = row[col];
  return typeof v === "number" ? v.toFixed(2) + "×" : "—";
}

function cellGain(row: Record<string, unknown>, col: string): string {
  const v = row[`${col}_gain`];
  return fmtGain(typeof v === "number" ? v : null);
}

type SortKey = "backend" | "method" | "model" | "correct" | "fast1" | "mean" | "median" | "best";

export default function ExperimentKernelSpeedups() {
  const [cohort, setCohort] = useCanvasState<Cohort>("cohort", "sample19");
  const [modelFilter, setModelFilter] = useCanvasState("modelFilter", "all");
  const [backendFilter, setBackendFilter] = useCanvasState("backendFilter", "all");
  const [methodFilter, setMethodFilter] = useCanvasState("methodFilter", "all");
  const [typeFilter, setTypeFilter] = useCanvasState("typeFilter", "all");
  const [sortKey, setSortKey] = useCanvasState<SortKey>("sortKey", "fast1");
  const [maxDepth, setMaxDepth] = useCanvasState<number>("maxDepth", ABS_MAX_DEPTH);
  const [excludePrecisionCheats, setExcludePrecisionCheats] = useCanvasState<boolean>(
    "excludePrecisionCheats",
    false,
  );

  const typedExps = EXPS.map((e) => summarizeExp(e, cohort, typeFilter, maxDepth, excludePrecisionCheats));

  const filteredExps = typedExps
    .filter((e) => {
      if (modelFilter !== "all" && e.model !== modelFilter) return false;
      if (backendFilter !== "all" && e.backend !== backendFilter) return false;
      if (methodFilter !== "all" && e.method !== methodFilter) return false;
      // Hide setups with no nodes yet for the active cohort (new-8 / all partial runs).
      if ((cohort === "new8" || cohort === "all") && e.attempted_n === 0) return false;
      return true;
    })
    .slice()
    .sort((a, b) => {
      const pick = (e: (typeof typedExps)[number]) => {
        if (sortKey === "backend") return e.backend;
        if (sortKey === "method") return e.method;
        if (sortKey === "model") return e.model;
        if (sortKey === "correct") return e.correct_pct;
        if (sortKey === "fast1") return e.fast1_pct;
        if (sortKey === "mean") return e.mean_speedup ?? -1;
        if (sortKey === "median") return e.median_speedup ?? -1;
        return e.best_speedup ?? -1;
      };
      const av = pick(a);
      const bv = pick(b);
      if (typeof av === "string" && typeof bv === "string") return av.localeCompare(bv);
      return (bv as number) - (av as number);
    });

  const cols = filteredExps.map((e) => e.col);
  const targetN = cohortIds(cohort).length;
  const matrix = buildMatrix(cohort, typeFilter, maxDepth, EXPS.map((e) => e.col), excludePrecisionCheats);

  const summaryRows = filteredExps.map((e) => [
    e.backend,
    e.method,
    e.model,
    `${e.correct_n}/${e.target_n}`,
    fmtPct(e.correct_pct),
    `${e.fast1_n}/${e.target_n}`,
    fmtPct(e.fast1_pct),
    fmtSp(e.mean_speedup),
    fmtSp(e.median_speedup),
    fmtSp(e.best_speedup),
    fmtGain(e.mean_gain),
  ]);

  const summaryTone = filteredExps.map((e) =>
    e.correct_pct < 50 ? ("danger" as const) : e.correct_n < e.target_n ? ("warning" as const) : ("success" as const),
  );

  const matrixFiltered =
    typeFilter === "all" ? matrix : matrix.filter((r) => r.ktype === typeFilter);
  const kernelTableRows = matrixFiltered.map((row) => [
    String(row.id),
    String(row.ktype),
    String(row.name ?? "—"),
    ...cols.flatMap((c) => [cellSp(row, c), cellGain(row, c)]),
    fmtGain(typeof row.best_gain === "number" ? row.best_gain : null),
  ]);

  const chartData = filteredExps.map((e) => ({ label: e.col, correct: e.correct_pct, fast1: e.fast1_pct }));

  const topFast1 = filteredExps.length ? Math.max(...filteredExps.map((e) => e.fast1_pct)) : 0;
  const topMean = filteredExps.length ? Math.max(...filteredExps.map((e) => e.mean_speedup ?? 0)) : 0;
  const topBest = filteredExps.length ? Math.max(...filteredExps.map((e) => e.best_speedup ?? 0)) : 0;
  const filteredTokens = sumTokenBuckets(
    filteredExps.map((e) =>
      e.tok_total == null
        ? null
        : { input: e.tok_in ?? 0, output: e.tok_out ?? 0, total: e.tok_total, calls: 0 },
    ),
  );

  return (
    <Stack gap={20} style={{ padding: 24 }}>
      <Stack gap={6}>
        <H1>Cutegen experiment scoreboard</H1>
        <Text tone="secondary">
          Speedup = ref / best correct gen time at depths 0–{maxDepth}. Correct = ≥1 passing instance in range.
          Fast1 = correct and speedup &gt; 1. Fractions use cohort size (19, 8, or 27). Token counts from
          TOKEN_USAGE CSVs (input / output / total), filtered to the selected cohort kernels. Updated {GENERATED}.
          {excludePrecisionCheats
            ? " I/O precision cheats excluded: k33 Triton/nopf/S, k49 CuTe/nopf/S counted as incorrect (no speedup in aggregates)."
            : ""}
        </Text>
      </Stack>

      {excludePrecisionCheats ? (
        <Callout tone="warning" title="I/O precision cheats excluded">
          <Stack gap={4}>
            {PRECISION_CHEATS.map((c) => (
              <Text key={`${c.col}-${c.kernel}`} size="sm">
                <Code>k{c.kernel}</Code> · <Code>{c.col}</Code> — {c.note}
              </Text>
            ))}
          </Stack>
        </Callout>
      ) : null}

      <Callout tone="info" title="Profiling modes">
        <Text size="sm">
          <Code>level1-profiled</Code> and <Code>level1-profiled-sonnet5</Code> = delayed profiling.
          <Code>level1-profiled-from-start-*</Code> and <Code>level1_from_start</Code> = profiling from start.
          <Code>*-minimal*</Code> = minimal-prompt ablation (nopf or from-start).
        </Text>
      </Callout>

      <Row gap={12} wrap>
        <Select
          label="Max depth"
          value={String(maxDepth)}
          onChange={(v) => setMaxDepth(Number(v))}
          options={maxDepthOptions()}
        />
        <Select
          value={cohort}
          onChange={(v) => setCohort(v as Cohort)}
          options={[
            { value: "sample19", label: "Cohort: Sample 19 kernels" },
            { value: "new8", label: "Cohort: New 8 kernels" },
            { value: "all", label: "Cohort: All 27 kernels" },
          ]}
        />
        <Select
          value={modelFilter}
          onChange={setModelFilter}
          options={[
            { value: "all", label: "Model: All" },
            { value: "Kimi K3", label: "Model: Kimi K3" },
            { value: "Sonnet 5", label: "Model: Sonnet 5" },
            { value: "GPT-5", label: "Model: GPT-5" },
          ]}
        />
        <Select
          value={backendFilter}
          onChange={setBackendFilter}
          options={[
            { value: "all", label: "Backend: All" },
            { value: "CUDA", label: "Backend: CUDA" },
            { value: "CuTe", label: "Backend: CuTe" },
            { value: "PTX", label: "Backend: PTX" },
            { value: "Triton", label: "Backend: Triton" },
          ]}
        />
        <Select
          value={methodFilter}
          onChange={setMethodFilter}
          options={[
            { value: "all", label: "Method: All" },
            { value: "No profiling", label: "Method: No profiling" },
            { value: "Delayed profiling", label: "Method: Delayed profiling" },
            { value: "Profiling from start", label: "Method: Profiling from start" },
            { value: "Minimal", label: "Method: Minimal" },
            { value: "Minimal from start", label: "Method: Minimal from start" },
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
        <Select
          value={sortKey}
          onChange={(v) => setSortKey(v as SortKey)}
          options={[
            { value: "fast1", label: "Sort: Fast1 %" },
            { value: "correct", label: "Sort: Correct %" },
            { value: "mean", label: "Sort: Mean speedup" },
            { value: "best", label: "Sort: Best speedup" },
          ]}
        />
        <Row gap={8} style={{ alignItems: "center" }}>
          <Toggle checked={excludePrecisionCheats} onChange={setExcludePrecisionCheats} />
          <Text size="sm">Exclude I/O precision cheats (k33, k49)</Text>
        </Row>
      </Row>

      <Grid columns={4} gap={12}>
        <Stat value={fmtPct(topFast1)} label="Highest Fast1 %" tone="success" />
        <Stat value={fmtSp(topMean)} label="Highest mean speedup" tone="info" />
        <Stat value={fmtSp(topBest)} label="Highest peak speedup" />
        <Stat value={String(targetN)} label={`Cohort size (${targetN})`} />
      </Grid>
      <Grid columns={3} gap={12}>
        <Stat value={fmtTokens(filteredTokens?.input)} label="Input tokens (filtered setups)" />
        <Stat value={fmtTokens(filteredTokens?.output)} label="Output tokens (filtered setups)" />
        <Stat value={fmtTokens(filteredTokens?.total)} label="Total tokens (filtered setups)" tone="info" />
      </Grid>

      <Stack gap={8}>
        <H2>Correct % vs Fast1 % ({cohortLabel(cohort)})</H2>
        <Text size="small" tone="secondary">
          Source: saved_nodes/* · depths 0–{maxDepth} · denominator = cohort size ({targetN})
        </Text>
        <BarChart
          categories={chartData.map((d) => d.label)}
          series={[
            { name: "Correct %", data: chartData.map((d) => d.correct), tone: "info" },
            { name: "Fast1 %", data: chartData.map((d) => d.fast1), tone: "success" },
          ]}
          valueSuffix="%"
          height={280}
        />
      </Stack>

      <Stack gap={8}>
        <H2>Experiment summary (depths 0–{maxDepth})</H2>
        <Table
          headers={["Backend", "Method", "Model", "Path", "Correct", "Correct %", "Fast1", "Fast1 %", "Mean", "Median", "Best", "Mean max/init", "In tok", "Out tok", "Total tok"]}
          rows={filteredExps.map((e) => [
            e.backend,
            e.method,
            e.model,
            e.path,
            `${e.correct_n}/${e.target_n}`,
            fmtPct(e.correct_pct),
            `${e.fast1_n}/${e.target_n}`,
            fmtPct(e.fast1_pct),
            fmtSp(e.mean_speedup),
            fmtSp(e.median_speedup),
            fmtSp(e.best_speedup),
            fmtGain(e.mean_gain),
            fmtTokens(e.tok_in),
            fmtTokens(e.tok_out),
            fmtTokens(e.tok_total),
          ])}
          rowTone={summaryTone}
          columnAlign={["left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"]}
          striped
          stickyHeader
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Per-kernel best speedup (depths 0–{maxDepth})</H2>
        <Text size="small" tone="secondary">
          Init = depth-0 speedup · max/init = best ÷ init · — = no correct instance in range for that kernel.
        </Text>
        {cols.length > 0 ? (
          <Table
            headers={[
              "ID",
              "Type",
              "Kernel",
              ...filteredExps.flatMap((e) => [`${e.col} best`, `${e.col} max/init`]),
              "Best max/init",
            ]}
            rows={kernelTableRows}
            striped
            stickyHeader
          />
        ) : null}
      </Stack>

      <Text size="small" tone="tertiary">
        {filteredExps.length} experiments · depths 0–{maxDepth} · refreshed {GENERATED}
      </Text>
    </Stack>
  );
}
'''


def scoreboard_header(payload: dict) -> str:
    return f'''import {{
  BarChart,
  Callout,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  Toggle,
  useCanvasState,
}} from "cursor/canvas";
'''

def speedup_header(payload: dict, extra_types: str) -> str:
    return f'''import {{
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  LineChart,
  Row,
  Select,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasState,
  useHostTheme,
}} from "cursor/canvas";

{extra_types}
{TOKEN_TYPES}
const GENERATED = {json.dumps(payload["generated"])};
const DEPTH_LABELS: string[] = {js(payload["depthLabels"])};
const MAX_DEPTH = {payload["maxDepth"]};
const SAMPLE19_IDS = {js(payload["sample19Ids"])};
const NEW8_IDS = {js(payload["new8Ids"])};
const KERNEL_TYPES: string[] = {js(payload["kernelTypes"])};
const EXPS = {js(payload["exps"])};
const KERNELS = {js(payload["kernels"])};
const CURVES: Record<string, Record<string, (number | null)[]>> = {js(payload["curves"])};
const TOKENS: Record<string, TokenAgg> = {js(payload.get("tokens") or {})};
type PrecisionCheat = {{ col: string; kernel: number; path: string; note: string }};
const PRECISION_CHEATS: PrecisionCheat[] = {js(payload.get("precisionCheats") or [])};
{SHARED_HELPERS}
'''

def data_header(payload: dict) -> str:
    return (
        scoreboard_header(payload)
        + TOKEN_TYPES
        + f'''
const GENERATED = {json.dumps(payload["generated"])};
const DEPTH_LABELS: string[] = {js(payload["depthLabels"])};
const MAX_DEPTH = {payload["maxDepth"]};
const SAMPLE19_IDS = {js(payload["sample19Ids"])};
const NEW8_IDS = {js(payload["new8Ids"])};
const KERNEL_TYPES: string[] = {js(payload["kernelTypes"])};
const EXPS = {js(payload["exps"])};
const KERNELS = {js(payload["kernels"])};
const CURVES: Record<string, Record<string, (number | null)[]>> = {js(payload["curves"])};
const TOKENS: Record<string, TokenAgg> = {js(payload.get("tokens") or {})};
type PrecisionCheat = {{ col: string; kernel: number; path: string; note: string }};
const PRECISION_CHEATS: PrecisionCheat[] = {js(payload.get("precisionCheats") or [])};
'''
        + SHARED_HELPERS
    )


def main() -> None:
    payload_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cutegen_canvas_payload.json")
    payload = json.loads(payload_path.read_text())

    speedup_tsx = speedup_header(
        payload,
        'type KernelMeta = { id: number; name: string; ktype: string; cohort: string };\n'
        'type ExpMeta = { col: string; path: string; label: string; backend: string; method: string; model: string; leaf: string };\n',
    ) + SPEEDUP_DEPTH_BODY

    scoreboard_tsx = data_header(payload) + SCOREBOARD_BODY

    CANVAS_DIR.mkdir(parents=True, exist_ok=True)
    (CANVAS_DIR / "speedup-vs-depth.canvas.tsx").write_text(speedup_tsx)
    (CANVAS_DIR / "model-experiment-speedups.canvas.tsx").write_text(scoreboard_tsx)
    print(f"wrote {CANVAS_DIR / 'speedup-vs-depth.canvas.tsx'} ({len(speedup_tsx)} chars)")
    print(f"wrote {CANVAS_DIR / 'model-experiment-speedups.canvas.tsx'} ({len(scoreboard_tsx)} chars)")


if __name__ == "__main__":
    main()
