/**
 * Daily activity history, kept from the extension's own samples.
 *
 * Owns folding one sample into per UTC day buckets and trimming to 30 days. It does not own
 * where the buckets are stored; the extension keeps them in its globalState.
 */

export interface DayBucket {
  /** UTC day as `YYYY-MM-DD`. */
  day: string;
  gpuSum: number;
  gpuCount: number;
  /** Quota spent that day per alias, in the alias's unit. */
  spent: Record<string, number>;
  /** Last remaining amount seen per alias, to measure the next drop. */
  lastRemaining: Record<string, number>;
}

export const HISTORY_DAYS = 30;

export function dayOf(seconds: number): string {
  return new Date(seconds * 1000).toISOString().slice(0, 10);
}

export function recordSample(
  history: DayBucket[],
  now: number,
  gpuMeanPercent: number | null,
  remaining: Record<string, number>,
): DayBucket[] {
  const day = dayOf(now);
  const buckets = history.map((b) => ({ ...b, spent: { ...b.spent }, lastRemaining: { ...b.lastRemaining } }));
  const previous = buckets[buckets.length - 1];
  let bucket = buckets.find((b) => b.day === day);
  if (!bucket) {
    bucket = { day, gpuSum: 0, gpuCount: 0, spent: {}, lastRemaining: { ...(previous?.lastRemaining ?? {}) } };
    buckets.push(bucket);
  }
  if (gpuMeanPercent !== null) {
    bucket.gpuSum += gpuMeanPercent;
    bucket.gpuCount += 1;
  }
  for (const [alias, value] of Object.entries(remaining)) {
    const last = bucket.lastRemaining[alias];
    // A rise is a top up or a reset, not negative spending.
    if (last !== undefined && value < last) {
      bucket.spent[alias] = (bucket.spent[alias] ?? 0) + (last - value);
    }
    bucket.lastRemaining[alias] = value;
  }
  const cutoff = dayOf(now - (HISTORY_DAYS - 1) * 86400);
  return buckets.filter((b) => b.day >= cutoff).sort((a, b) => a.day.localeCompare(b.day));
}

export function gpuMean(bucket: DayBucket): number | null {
  return bucket.gpuCount ? bucket.gpuSum / bucket.gpuCount : null;
}
