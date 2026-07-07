import { useEffect, useMemo, useRef, useState } from "react";

import type { MonitoringPayload, MonitoringSystemResponse, JobListResponse } from "@/api/types";
import { api } from "@/api/client";

type MonitoringState = {
  system: MonitoringSystemResponse | null;
  activeJobs: JobListResponse["items"];
  logs: string[];
  connected: boolean;
  fallback: boolean;
};

export function useMonitoringFeed(): MonitoringState {
  const [system, setSystem] = useState<MonitoringSystemResponse | null>(null);
  const [activeJobs, setActiveJobs] = useState<JobListResponse["items"]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [fallback, setFallback] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  const loadFallback = async () => {
    try {
      const [systemValue, jobsValue, logsValue] = await Promise.all([
        api.monitoring.system(),
        api.monitoring.jobs(),
        api.monitoring.logs(),
      ]);
      setSystem(systemValue);
      setActiveJobs(jobsValue.items);
      setLogs(logsValue.items);
    } catch {
      // keep previous values when backend is not yet available
    }
  };

  useEffect(() => {
    let closed = false;
    const socket = new WebSocket(api.monitoring.wsUrl());
    socket.onopen = () => {
      if (closed) return;
      setConnected(true);
      setFallback(false);
      if (timer.current) window.clearInterval(timer.current);
    };
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data as string) as MonitoringPayload;
      setSystem(payload.system);
      setActiveJobs(payload.active_jobs);
    };
    socket.onerror = () => {
      if (closed) return;
      setConnected(false);
      setFallback(true);
      void loadFallback();
      timer.current = window.setInterval(() => {
        void loadFallback();
      }, 2500);
    };
    socket.onclose = () => {
      if (closed) return;
      setConnected(false);
      setFallback(true);
      void loadFallback();
      timer.current = window.setInterval(() => {
        void loadFallback();
      }, 2500);
    };
    void loadFallback();
    return () => {
      closed = true;
      socket.close();
      if (timer.current) window.clearInterval(timer.current);
    };
  }, []);

  return useMemo(
    () => ({ system, activeJobs, logs, connected, fallback }),
    [system, activeJobs, logs, connected, fallback],
  );
}
