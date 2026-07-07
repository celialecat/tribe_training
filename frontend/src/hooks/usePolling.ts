import { useEffect, useRef, useState } from "react";

export function usePolling<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[],
  intervalMs = 2000,
): { data: T | null; loading: boolean; error: string | null; refresh: () => Promise<void> } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const active = useRef(true);

  useEffect(() => {
    active.current = true;

    const run = async () => {
      try {
        const value = await loader();
        if (!active.current) return;
        setData(value);
        setError(null);
      } catch (err) {
        if (!active.current) return;
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        if (active.current) setLoading(false);
      }
    };

    void run();
    const timer = window.setInterval(() => {
      void run();
    }, intervalMs);

    return () => {
      active.current = false;
      if (timer) window.clearInterval(timer);
    };
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps

  const refresh = async () => {
    setLoading(true);
    try {
      const value = await loader();
      setData(value);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return { data, loading, error, refresh };
}
