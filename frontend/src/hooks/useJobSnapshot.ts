import { useMemo } from "react";

import { usePolling } from "./usePolling";

export function useJobSnapshot<T>(
  jobId: string | null,
  loader: (jobId: string) => Promise<T>,
  intervalMs = 2000,
): { data: T | null; loading: boolean; error: string | null; refresh: () => Promise<void> } {
  const deps = useMemo(() => [jobId] as const, [jobId]);
  return usePolling(() => {
    if (!jobId) {
      return Promise.resolve(null as T | null);
    }
    return loader(jobId);
  }, deps, intervalMs);
}
