import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActionIcon,
  Badge,
  Button,
  Card,
  Divider,
  Group,
  MultiSelect,
  NumberInput,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Tabs,
  TagsInput,
  Text,
  TextInput,
  Textarea,
} from "@mantine/core";
import { DatePickerInput } from "@mantine/dates";
import { IconDownload, IconDatabase, IconPlus, IconRefresh, IconTrash } from "@tabler/icons-react";

import { api, downloadBlob } from "@/api/client";
import type { AnalysisMethodResponse, AnalysisRunResult, DatasetFilter, DatasetPreviewResponse, TensorListResponse } from "@/api/types";
import { ChartCard, SimpleBarChart, SimpleLineChart } from "@/components/charts";
import { ChartEmpty, EmptyState, ErrorState, LoadingState, MetricTiles, SectionCard, StatusBadge, SubtleHint } from "@/components/ui";
import { useJobSnapshot } from "@/hooks/useJobSnapshot";
import { useMonitoringFeed } from "@/hooks/useMonitoringFeed";
import { usePolling } from "@/hooks/usePolling";
import { formatBytes, formatNumber } from "@/utils/format";

type TimeWindowMode = "mean" | "index" | "window";
type ExportFormat = "csv" | "npz" | "pt";

type AnalysisControls = {
  method: string;
  n_components: number;
  target: string;
  metrics: string[];
  region_names: string[];
  normalization: "none" | "zscore";
  time_window_mode: TimeWindowMode;
  time_index: number;
  time_start: number;
  time_end: number;
  screening_threshold: number;
  top_k: number;
  n_slices: number;
};

const DEFAULT_CONTROLS: AnalysisControls = {
  method: "pca",
  n_components: 3,
  target: "log_likes",
  metrics: ["likes", "views", "comments", "engagement"],
  region_names: [],
  normalization: "none",
  time_window_mode: "mean",
  time_index: 0,
  time_start: 0,
  time_end: 5,
  screening_threshold: 0.2,
  top_k: 10,
  n_slices: 10,
};

type DateRangeValue = [Date | null, Date | null];

export function ScientificDashboard() {
  const modes = usePolling(() => api.datasets.modes(), [], 60000);
  const tensors = usePolling<TensorListResponse>(() => api.tensors.list(), [], 8000);
  const tribe = usePolling(() => api.tribe.status(), [], 4000);
  const methods = usePolling<AnalysisMethodResponse[]>(() => api.analysis.methods(), [], 60000);
  const evaluationLatest = usePolling(() => api.evaluation.latest(), [], 10000);
  const monitoring = useMonitoringFeed();

  const [mode, setMode] = useState("");
  const [channelUrl, setChannelUrl] = useState("");
  const [filter, setFilter] = useState<DatasetFilter>({});
  const [dateRange, setDateRange] = useState<DateRangeValue>([null, null]);
  const [preview, setPreview] = useState<DatasetPreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Record<string, string | null>>({
    download: null,
    validate: null,
    tribe: null,
    training: null,
    pipeline: null,
  });
  const [pipelineMode, setPipelineMode] = useState("");
  const [pipelineCount, setPipelineCount] = useState(12);
  const [analysisControls, setAnalysisControls] = useState<AnalysisControls>(DEFAULT_CONTROLS);
  const [analysis, setAnalysis] = useState<AnalysisRunResult | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const trainingJob = useJobSnapshot(jobs.training, api.training.job, 2000);
  const pipelineJob = useJobSnapshot(jobs.pipeline, api.pipeline.job, 2000);
  const analysisDependencyKey = useMemo(
    () =>
      [
        analysisControls.method,
        analysisControls.n_components,
        analysisControls.target,
        analysisControls.metrics.join(","),
        analysisControls.region_names.join(","),
        analysisControls.normalization,
        analysisControls.time_window_mode,
        analysisControls.time_index,
        analysisControls.time_start,
        analysisControls.time_end,
        analysisControls.screening_threshold,
        analysisControls.top_k,
        analysisControls.n_slices,
      ].join("|"),
    [analysisControls],
  );

  useEffect(() => {
    const availableModes = Object.keys(modes.data ?? {});
    if (!mode && availableModes.length > 0) {
      setMode(availableModes[0]);
      setPipelineMode(availableModes[0]);
    }
  }, [mode, modes.data]);

  useEffect(() => {
    if (dateRange[0]) {
      setFilter((current) => ({ ...current, date_from: dateRange[0]?.toISOString() ?? null }));
    }
    if (dateRange[1]) {
      setFilter((current) => ({ ...current, date_to: dateRange[1]?.toISOString() ?? null }));
    }
  }, [dateRange]);

  const modeEntries = useMemo(() => Object.entries(modes.data ?? {}), [modes.data]);
  const selectedMode = mode ? (modes.data?.[mode] ?? null) : null;
  const tensorItems = tensors.data?.items ?? [];
  const analysisMethods = methods.data ?? [];
  const tribeState = tribe.data;

  async function previewDataset() {
    if (!mode) return;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      setPreview(
        await api.datasets.preview({
          mode,
          filter,
          count: 100,
        }),
      );
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : "Preview failed");
    } finally {
      setPreviewLoading(false);
    }
  }

  async function addChannel() {
    if (!mode || !channelUrl.trim()) return;
    await api.datasets.addChannel(mode, channelUrl.trim());
    setChannelUrl("");
    await modes.refresh();
  }

  async function removeChannel(url: string) {
    if (!mode) return;
    await api.datasets.removeChannel(mode, url);
    await modes.refresh();
  }

  async function runDatasetBuild() {
    if (!mode) return;
    const response = await api.datasets.build({ mode, filter, count: 100 });
    setJobs((current) => ({ ...current, download: response.job_id }));
  }

  async function runDatasetValidate() {
    const response = await api.datasets.validate({ filter, count: 100 });
    setJobs((current) => ({ ...current, validate: response.job_id }));
  }

  async function runTribe() {
    const response = await api.tribe.run({ limit: 100 });
    setJobs((current) => ({ ...current, tribe: response.job_id }));
  }

  async function runTraining() {
    const response = await api.training.start({});
    setJobs((current) => ({ ...current, training: response.job_id }));
  }

  async function runEvaluation() {
    const response = await api.evaluation.run();
    setAnalysis({
      method: "evaluation",
      ...response,
    });
  }

  async function runPipeline() {
    if (!pipelineMode) return;
    const response = await api.pipeline.run({ mode: pipelineMode, n_videos: pipelineCount });
    setJobs((current) => ({ ...current, pipeline: response.job_id }));
  }

  const runAnalysis = useCallback(async () => {
    if (!methods.data?.length) return;
    setAnalysisLoading(true);
    setAnalysisError(null);
    try {
      const result = await api.analysis.run({
        method: analysisControls.method,
        n_components: analysisControls.n_components,
        target: analysisControls.target,
        metrics: analysisControls.metrics,
        region_names: analysisControls.region_names,
        normalization: analysisControls.normalization,
        screening_threshold: analysisControls.screening_threshold,
        top_k: analysisControls.top_k,
        n_slices: analysisControls.n_slices,
        time_window: {
          mode: analysisControls.time_window_mode,
          index: analysisControls.time_index,
          start: analysisControls.time_start,
          end: analysisControls.time_end,
        },
      });
      setAnalysis(result);
    } catch (error) {
      setAnalysis(null);
      setAnalysisError(error instanceof Error ? error.message : "Analysis failed");
    } finally {
      setAnalysisLoading(false);
    }
  }, [analysisControls, methods.data]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void runAnalysis();
    }, 400);
    return () => window.clearTimeout(timer);
  }, [analysisDependencyKey, runAnalysis]);

  async function exportAnalysis(fmt: ExportFormat) {
    const blob = await api.analysis.export(
      {
        method: analysisControls.method,
        n_components: analysisControls.n_components,
        target: analysisControls.target,
        metrics: analysisControls.metrics,
        region_names: analysisControls.region_names,
        normalization: analysisControls.normalization,
        screening_threshold: analysisControls.screening_threshold,
        top_k: analysisControls.top_k,
        n_slices: analysisControls.n_slices,
        time_window: {
          mode: analysisControls.time_window_mode,
          index: analysisControls.time_index,
          start: analysisControls.time_start,
          end: analysisControls.time_end,
        },
      },
      fmt,
    );
    downloadBlob(blob, `analysis-${analysisControls.method}.${fmt}`);
  }

  return (
    <Stack gap="lg">
      <SectionCard
        title="Dataset Builder"
        description="Edit channel sources, preview the expected dataset, and keep the empty-state clean when the local DB has no videos."
        action={
          <Button variant="light" leftSection={<IconRefresh size={16} />} onClick={() => void previewDataset()}>
            Preview
          </Button>
        }
      >
        {modes.loading ? (
          <LoadingState label="Loading dataset modes…" />
        ) : modes.error ? (
          <ErrorState error={modes.error} />
        ) : modeEntries.length === 0 ? (
          <EmptyState title="No dataset modes" description="The backend returned no modes yet." />
        ) : (
          <Stack gap="md">
            <Group align="flex-end" grow>
              <Select
                label="Mode"
                data={modeEntries.map(([value]) => ({ value, label: value }))}
                value={mode}
                onChange={(value) => setMode(value ?? "")}
              />
              <TextInput
                label="Add channel URL"
                placeholder="https://www.youtube.com/@..."
                value={channelUrl}
                onChange={(event) => setChannelUrl(event.currentTarget.value)}
              />
              <Button onClick={() => void addChannel()} leftSection={<IconPlus size={16} />}>
                Add
              </Button>
            </Group>

            <Paper withBorder radius="lg" p="md">
              <Group justify="space-between" mb="xs">
                <Text fw={600}>Channels</Text>
                <Badge variant="light">{selectedMode?.channels?.length ?? 0} configured</Badge>
              </Group>
              {selectedMode?.channels?.length ? (
                <Stack gap="xs">
                  {selectedMode.channels.map((url) => (
                    <Paper key={url} withBorder radius="md" p="sm">
                      <Group justify="space-between">
                        <Text size="sm">{url}</Text>
                        <ActionIcon variant="subtle" color="red" onClick={() => void removeChannel(url)}>
                          <IconTrash size={16} />
                        </ActionIcon>
                      </Group>
                    </Paper>
                  ))}
                </Stack>
              ) : (
                <EmptyState title="No channels yet" description="Add a channel URL to begin building the dataset." icon={<IconDatabase size={28} />} />
              )}
            </Paper>

            <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
              <NumberInput
                label="Min duration (sec)"
                value={filter.min_duration_seconds ?? undefined}
                onChange={(value) => setFilter((current) => ({ ...current, min_duration_seconds: Number(value) || null }))}
              />
              <NumberInput
                label="Max duration (sec)"
                value={filter.max_duration_seconds ?? undefined}
                onChange={(value) => setFilter((current) => ({ ...current, max_duration_seconds: Number(value) || null }))}
              />
              <Select
                label="Language"
                data={[
                  { value: "", label: "Any" },
                  { value: "en", label: "English" },
                  { value: "es", label: "Spanish" },
                  { value: "fr", label: "French" },
                ]}
                value={filter.language ?? ""}
                onChange={(value) => setFilter((current) => ({ ...current, language: value || null }))}
              />
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
              <NumberInput
                label="Min likes"
                value={filter.min_likes ?? undefined}
                onChange={(value) => setFilter((current) => ({ ...current, min_likes: Number(value) || null }))}
              />
              <NumberInput
                label="Max likes"
                value={filter.max_likes ?? undefined}
                onChange={(value) => setFilter((current) => ({ ...current, max_likes: Number(value) || null }))}
              />
              <TagsInput
                label="Categories"
                placeholder="education, science"
                value={filter.categories ?? []}
                onChange={(value) => setFilter((current) => ({ ...current, categories: value }))}
              />
            </SimpleGrid>

            <DatePickerInput
              type="range"
              label="Publication date range"
              value={dateRange}
              onChange={setDateRange}
              clearable
            />

            <Group>
              <Button onClick={() => void runDatasetBuild()}>Download Dataset</Button>
              <Button variant="light" onClick={() => void runDatasetValidate()}>
                Validate Dataset
              </Button>
            </Group>

            {previewLoading ? (
              <LoadingState label="Generating preview…" />
            ) : previewError ? (
              <ErrorState error={previewError} />
            ) : preview ? (
              <SimpleGrid cols={{ base: 1, xl: 3 }} spacing="md">
                <MetricTiles
                  items={[
                    { label: "Estimated videos", value: formatNumber(numberValue(preview.estimated_count)) },
                    { label: "Estimated storage", value: formatBytes(numberValue(preview.estimated_storage_bytes)) },
                    { label: "Mode hint", value: selectedMode?.hint ?? "—" },
                  ]}
                />
                <ChartCard title="Publication distribution" height={260}>
                  <SimpleBarChart
                    data={histogramToSeries(preview.publication_date_histogram)}
                    xKey="bucket"
                    bars={["count"]}
                  />
                </ChartCard>
                <Paper withBorder radius="lg" p="md">
                  <Text fw={600} mb="sm">
                    Preview summary
                  </Text>
                  <Text size="sm" c="dimmed">
                    {selectedMode?.hint ?? "No mode hint returned."}
                  </Text>
                  <Divider my="sm" />
                  <Text size="sm">
                    Count: <strong>{formatNumber(numberValue(preview.estimated_count))}</strong>
                  </Text>
                  <Text size="sm">
                    Storage: <strong>{formatBytes(numberValue(preview.estimated_storage_bytes))}</strong>
                  </Text>
                </Paper>
              </SimpleGrid>
            ) : (
              <SubtleHint text="Press Preview to estimate count, storage, and publication distribution before downloading." />
            )}
          </Stack>
        )}
      </SectionCard>

      <Tabs defaultValue="workflow">
        <Tabs.List>
          <Tabs.Tab value="workflow">Scientific workflow</Tabs.Tab>
          <Tabs.Tab value="pipeline">Pipeline</Tabs.Tab>
          <Tabs.Tab value="monitoring">Monitoring</Tabs.Tab>
          <Tabs.Tab value="analysis">Latent analysis</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="workflow" pt="md">
          <Stack gap="md">
            <SimpleGrid cols={{ base: 1, xl: 3 }} spacing="md">
              <Card withBorder radius="lg" p="md">
                <Group justify="space-between">
                  <Text fw={600}>Download dataset</Text>
                  <Button size="xs" onClick={() => void runDatasetBuild()}>
                    Run
                  </Button>
                </Group>
                <Text size="sm" c="dimmed" mt="xs">
                  {jobs.download ?? "No dataset job yet."}
                </Text>
              </Card>
              <Card withBorder radius="lg" p="md">
                <Group justify="space-between">
                  <Text fw={600}>Validate dataset</Text>
                  <Button size="xs" variant="light" onClick={() => void runDatasetValidate()}>
                    Run
                  </Button>
                </Group>
                <Text size="sm" c="dimmed" mt="xs">
                  {jobs.validate ?? "No validation job yet."}
                </Text>
              </Card>
              <Card withBorder radius="lg" p="md">
                <Group justify="space-between">
                  <Text fw={600}>Run TRIBE v2</Text>
                  <Button size="xs" variant="light" onClick={() => void runTribe()}>
                    Run
                  </Button>
                </Group>
                <Text size="sm" c="dimmed" mt="xs">
                  {tribeState ? `${tribeState.cached} cached / ${tribeState.remaining} remaining` : "No TRIBE status yet."}
                </Text>
              </Card>
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
              <SectionCard title="Brain tensors" description="Tensor dimensions, cache count, and storage usage.">
                {tensors.loading ? (
                  <LoadingState label="Loading tensors…" />
                ) : tensorItems.length === 0 ? (
                  <EmptyState title="No cached tensors" description="The local database is empty, so there are no cached brain tensors yet." />
                ) : (
                  <Stack gap="md">
                    <MetricTiles
                      items={[
                        { label: "Cached tensors", value: formatNumber(numberValue(tensors.data?.count)) },
                        { label: "Storage", value: formatBytes(numberValue(tensors.data?.storage_bytes)) },
                        { label: "First shape", value: tensorItems[0] ? tensorShape(tensorItems[0]) : "—" },
                        { label: "Items", value: formatNumber(tensorItems.length) },
                      ]}
                    />
                    <Paper withBorder radius="lg" p="md">
                      <Stack gap="xs">
                        {tensorItems.slice(0, 5).map((item) => (
                          <Group key={item.video_id} justify="space-between">
                            <Text size="sm">Video {item.video_id}</Text>
                            <Text size="sm" c="dimmed">
                              {tensorShape(item)} · {formatBytes(item.file_size_bytes)}
                            </Text>
                          </Group>
                        ))}
                      </Stack>
                    </Paper>
                  </Stack>
                )}
              </SectionCard>

              <SectionCard title="Train and evaluate" description="Run the encoder, then inspect the evaluation metrics.">
                <Stack gap="md">
                  <Group justify="space-between">
                    <Button size="sm" onClick={() => void runTraining()}>
                      Train brain encoder
                    </Button>
                    <Button size="sm" variant="light" onClick={() => void runEvaluation()}>
                      Evaluate model
                    </Button>
                  </Group>
                  <Text size="sm" c="dimmed">
                    {trainingJob.data ? `Status: ${trainingJob.data.status}` : jobs.training ?? "No training job selected."}
                  </Text>
                  <Text size="sm" c="dimmed">
                    {typeof evaluationLatest.data?.n_samples === "number"
                      ? `${evaluationLatest.data.n_samples} samples evaluated`
                      : "No evaluation yet."}
                  </Text>
                </Stack>
              </SectionCard>
            </SimpleGrid>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="pipeline" pt="md">
          <SectionCard title="Run full pipeline" description="Download → validate → TRIBE → cache → train → evaluate.">
            <Stack gap="md">
              <Group align="flex-end">
                <Select
                  label="Mode"
                  data={modeEntries.map(([value]) => ({ value, label: value }))}
                  value={pipelineMode}
                  onChange={(value) => setPipelineMode(value ?? "")}
                />
                <NumberInput label="Videos" value={pipelineCount} min={1} onChange={(value) => setPipelineCount(Number(value) || 1)} />
                <Button onClick={() => void runPipeline()}>Run pipeline</Button>
              </Group>
              {pipelineJob.data ? (
                <MetricTiles
                  items={[
                    { label: "Job", value: pipelineJob.data.id },
                    { label: "Stage", value: scalarValue(pipelineJob.data.progress?.stage) },
                    { label: "Current", value: scalarValue(pipelineJob.data.progress?.current) },
                    { label: "Status", value: pipelineJob.data.status },
                  ]}
                />
              ) : (
                <EmptyState title="No pipeline job" description="Choose a mode and start the full pipeline to see stage-by-stage progress." />
              )}
            </Stack>
          </SectionCard>
        </Tabs.Panel>

        <Tabs.Panel value="monitoring" pt="md">
          <Stack gap="md">
            <MetricTiles
              items={[
                { label: "CPU", value: monitoring.system ? `${monitoring.system.cpu_percent.toFixed(1)}%` : "—" },
                {
                  label: "RAM",
                  value: monitoring.system
                    ? `${formatBytes(monitoring.system.memory_used)} / ${formatBytes(monitoring.system.memory_total)}`
                    : "—",
                },
                {
                  label: "Disk",
                  value: monitoring.system
                    ? `${formatBytes(monitoring.system.disk_used)} / ${formatBytes(monitoring.system.disk_total)}`
                    : "—",
                },
                { label: "Connection", value: monitoring.connected ? "WebSocket" : monitoring.fallback ? "Polling" : "Connecting" },
              ]}
            />

            <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
              <SectionCard title="System usage" description="Live system telemetry from /monitoring/system or the websocket feed.">
                {monitoring.system ? (
                  <ChartCard title="Current usage" height={260}>
                    <SimpleBarChart
                      data={[
                        {
                          label: "Current",
                          cpu: monitoring.system.cpu_percent,
                          ram: (monitoring.system.memory_used / monitoring.system.memory_total) * 100,
                          disk: (monitoring.system.disk_used / monitoring.system.disk_total) * 100,
                        },
                      ]}
                      xKey="label"
                      bars={["cpu", "ram", "disk"]}
                    />
                  </ChartCard>
                ) : (
                  <EmptyState title="No system metrics yet" description="The feed starts empty and remains clean until the backend emits telemetry." />
                )}
              </SectionCard>

              <SectionCard title="Active jobs" description="Live jobs from the monitoring feed.">
                {monitoring.activeJobs.length === 0 ? (
                  <EmptyState title="No active jobs" description="Once a job starts, it will appear here with a status badge." />
                ) : (
                  <Stack gap="xs">
                    {monitoring.activeJobs.map((job) => (
                      <Paper key={job.id} withBorder radius="md" p="sm">
                        <Group justify="space-between">
                          <Stack gap={2}>
                            <Text fw={600}>{job.type}</Text>
                            <Text size="xs" c="dimmed">
                              {job.id}
                            </Text>
                          </Stack>
                          <StatusBadge status={job.status} />
                        </Group>
                      </Paper>
                    ))}
                  </Stack>
                )}
              </SectionCard>
            </SimpleGrid>

            <SectionCard title="Latest logs" description="Recent log lines from active jobs.">
              {monitoring.logs.length === 0 ? (
                <EmptyState title="No logs yet" description="Log lines appear here once jobs begin running." />
              ) : (
                <Paper withBorder radius="lg" p="md">
                  <Text ff="monospace" size="sm" style={{ whiteSpace: "pre-wrap" }}>
                    {monitoring.logs.slice(-40).join("\n")}
                  </Text>
                </Paper>
              )}
            </SectionCard>
          </Stack>
        </Tabs.Panel>

        <Tabs.Panel value="analysis" pt="md">
          <SectionCard
            title="Latent brain analysis"
            description="PCA, supervised PCA, PLS, CCA, and conditional-variance/SIR analysis."
            action={
              <Group>
                <Button variant="light" leftSection={<IconRefresh size={16} />} onClick={() => void runAnalysis()}>
                  Run
                </Button>
                <Button variant="light" leftSection={<IconDownload size={16} />} onClick={() => void exportAnalysis("csv")} disabled={!analysis}>
                  CSV
                </Button>
                <Button variant="light" leftSection={<IconDownload size={16} />} onClick={() => void exportAnalysis("npz")} disabled={!analysis}>
                  NumPy
                </Button>
                <Button variant="light" leftSection={<IconDownload size={16} />} onClick={() => void exportAnalysis("pt")} disabled={!analysis}>
                  PyTorch
                </Button>
              </Group>
            }
          >
            <Stack gap="md">
              {methods.loading ? (
                <LoadingState label="Loading methods…" />
              ) : methods.error ? (
                <ErrorState error={methods.error} />
              ) : (
                <Stack gap="md">
                  <Select
                    label="Method"
                    data={analysisMethods.map((item) => ({ value: item.name, label: item.name }))}
                    value={analysisControls.method}
                    onChange={(value) => setAnalysisControls((current) => ({ ...current, method: value ?? current.method }))}
                  />
                  <Text c="dimmed" size="sm">
                    {analysisMethods.find((item) => item.name === analysisControls.method)?.description}
                  </Text>
                  <Textarea
                    label="Conditional variance justification"
                    value={
                      typeof analysis?.method_justification === "string"
                        ? analysis.method_justification
                        : "SIR estimates a sufficient dimension reduction subspace. PCA is unsupervised and therefore not appropriate for conditional-variance objectives."
                    }
                    minRows={4}
                    readOnly
                  />
                  <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
                    <NumberInput
                      label="Components"
                      value={analysisControls.n_components}
                      min={1}
                      max={10}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, n_components: Number(value) || 1 }))}
                    />
                    <Select
                      label="Target"
                      data={[
                        { value: "log_likes", label: "Log likes" },
                        { value: "likes", label: "Likes" },
                        { value: "views", label: "Views" },
                        { value: "comments", label: "Comments" },
                        { value: "engagement", label: "Engagement" },
                        { value: "virality", label: "Virality" },
                      ]}
                      value={analysisControls.target}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, target: value ?? current.target }))}
                    />
                    <Select
                      label="Normalization"
                      data={[
                        { value: "none", label: "None" },
                        { value: "zscore", label: "Z-score" },
                      ]}
                      value={analysisControls.normalization}
                      onChange={(value) =>
                        setAnalysisControls((current) => ({
                          ...current,
                          normalization: (value as "none" | "zscore") ?? current.normalization,
                        }))
                      }
                    />
                  </SimpleGrid>
                  <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
                    <Select
                      label="Time window"
                      data={[
                        { value: "mean", label: "Average" },
                        { value: "index", label: "Single timestep" },
                        { value: "window", label: "Range average" },
                      ]}
                      value={analysisControls.time_window_mode}
                      onChange={(value) =>
                        setAnalysisControls((current) => ({
                          ...current,
                          time_window_mode: (value as TimeWindowMode) ?? current.time_window_mode,
                        }))
                      }
                    />
                    <NumberInput
                      label="Timestep index"
                      value={analysisControls.time_index}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, time_index: Number(value) || 0 }))}
                    />
                    <NumberInput
                      label="Window end"
                      value={analysisControls.time_end}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, time_end: Number(value) || 0 }))}
                    />
                  </SimpleGrid>
                  <TagsInput
                    label="Brain regions"
                    placeholder="Type region names"
                    value={analysisControls.region_names}
                    onChange={(value) => setAnalysisControls((current) => ({ ...current, region_names: value }))}
                  />
                  <MultiSelect
                    label="CCA metrics"
                    data={["likes", "views", "comments", "engagement", "virality"].map((metric) => ({ value: metric, label: metric }))}
                    value={analysisControls.metrics}
                    onChange={(value) => setAnalysisControls((current) => ({ ...current, metrics: value }))}
                  />
                  <SimpleGrid cols={{ base: 1, lg: 3 }} spacing="md">
                    <NumberInput
                      label="Screening threshold"
                      value={analysisControls.screening_threshold}
                      min={0}
                      max={1}
                      step={0.05}
                      onChange={(value) =>
                        setAnalysisControls((current) => ({ ...current, screening_threshold: Number(value) || 0 }))
                      }
                    />
                    <NumberInput
                      label="Top-k features"
                      value={analysisControls.top_k}
                      min={1}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, top_k: Number(value) || 1 }))}
                    />
                    <NumberInput
                      label="SIR slices"
                      value={analysisControls.n_slices}
                      min={2}
                      onChange={(value) => setAnalysisControls((current) => ({ ...current, n_slices: Number(value) || 2 }))}
                    />
                  </SimpleGrid>

                  <Group>
                    <Button onClick={() => void runAnalysis()}>Run analysis</Button>
                  </Group>

                  {analysisLoading ? (
                    <LoadingState label="Running latent analysis…" />
                  ) : analysisError ? (
                    <ErrorState error={analysisError} />
                  ) : analysis?.error ? (
                    <ErrorState title="Analysis unavailable" error={errorMessage(analysis.error)} />
                  ) : analysis ? (
                    <Stack gap="md">
                      <MetricTiles
                        items={[
                          { label: "Requested", value: String(analysisControls.n_components) },
                          { label: "Samples", value: String(numberValue(analysis.n_samples)) },
                          { label: "Features", value: String(numberValue(analysis.n_features)) },
                          { label: "Clamped", value: analysis.clamped ? "Yes" : "No" },
                        ]}
                      />
                      {Array.isArray(analysis.explained_variance_ratio) ? (
                        <ChartCard title="Scree plot" height={260}>
                          <SimpleLineChart
                            data={(analysis.explained_variance_ratio as number[]).map((value, index) => ({
                              component: index + 1,
                              variance: value,
                              cumulative: numberAt((analysis.cumulative_explained_variance as number[] | undefined)?.[index]),
                            }))}
                            xKey="component"
                            lines={["variance", "cumulative"]}
                          />
                        </ChartCard>
                      ) : null}
                      {Array.isArray(analysis.projections) ? (
                        <ChartCard title="Video projections" height={260}>
                          <SimpleBarChart
                            data={projectionSeries(analysis.projections)}
                            xKey="video"
                            bars={["component1", "component2"]}
                          />
                        </ChartCard>
                      ) : null}
                      {Array.isArray(analysis.components) ? (
                        <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
                          {(analysis.components as Array<Record<string, unknown>>).slice(0, 4).map((component, index) => (
                            <Card key={index} withBorder radius="lg" p="md">
                              <Group justify="space-between" mb="sm">
                                <Text fw={600}>Component {index + 1}</Text>
                                <Badge variant="light">{correlationText(component)}</Badge>
                              </Group>
                              {brainMap(component) ? (
                                <SimpleGrid cols={2} spacing="xs">
                                  <MiniHeatmap label="Left hemisphere" values={brainMap(component)!.left} />
                                  <MiniHeatmap label="Right hemisphere" values={brainMap(component)!.right} />
                                </SimpleGrid>
                              ) : (
                                <ChartEmpty title="No brain map" description="This component did not include a cortical heatmap." />
                              )}
                            </Card>
                          ))}
                        </SimpleGrid>
                      ) : null}
                    </Stack>
                  ) : (
                    <EmptyState title="No analysis result yet" description="The panel stays empty until a real backend response is available." />
                  )}
                </Stack>
              )}
            </Stack>
          </SectionCard>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}

function histogramToSeries(histogram: DatasetPreviewResponse["publication_date_histogram"]) {
  return Object.entries(histogram.by_month ?? {}).map(([bucket, count]) => ({ bucket, count }));
}

function projectionSeries(projections: unknown): Array<Record<string, number | string>> {
  if (!Array.isArray(projections)) return [];
  return projections.slice(0, 20).map((row, index) => {
    const items = Array.isArray(row) ? row : [];
    return {
      video: `V${index + 1}`,
      component1: numberAt(items[0]),
      component2: numberAt(items[1]),
    };
  });
}

function brainMap(component: Record<string, unknown>): { left: number[]; right: number[] } | null {
  const value = component.brain_map;
  if (!isRecord(value)) return null;
  const left = value.left;
  const right = value.right;
  if (!Array.isArray(left) || !Array.isArray(right)) return null;
  return { left: left.map(numberAt), right: right.map(numberAt) };
}

function MiniHeatmap(props: { label: string; values: number[] }) {
  const cells = downsample(props.values, 12, 12);
  return (
    <Stack gap={4}>
      <Text size="xs" c="dimmed">
        {props.label}
      </Text>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 2, minHeight: 120 }}>
        {cells.flat().map((value, index) => (
          <div key={`${props.label}-${index}`} style={{ aspectRatio: "1 / 1", borderRadius: 4, background: heatColor(value) }} />
        ))}
      </div>
    </Stack>
  );
}

function downsample(values: number[], rows: number, cols: number): number[][] {
  const total = rows * cols;
  const chunkSize = Math.max(1, Math.floor(values.length / total));
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => {
      const start = (row * cols + col) * chunkSize;
      const slice = values.slice(start, start + chunkSize);
      return slice.length ? slice.reduce((sum, value) => sum + value, 0) / slice.length : 0;
    }),
  );
}

function heatColor(value: number): string {
  const normalized = Math.max(-1, Math.min(1, value));
  const alpha = Math.abs(normalized) * 0.85 + 0.1;
  return normalized >= 0 ? `rgba(79, 70, 229, ${alpha})` : `rgba(244, 63, 94, ${alpha})`;
}

function tensorShape(item: { shape: number[]; num_timesteps: number | null; num_features: number | null }): string {
  const shape = item.shape.join(" × ");
  return item.num_timesteps ? `${shape} · ${item.num_timesteps} steps` : shape;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function numberAt(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function errorMessage(error: unknown): string {
  if (isRecord(error) && typeof error.message === "string") return error.message;
  return "No data available";
}

function correlationText(component: Record<string, unknown>): string {
  const correlations = component.correlations;
  if (!isRecord(correlations)) return "No correlations";
  const likes = numberValue(correlations.likes);
  return `likes ${likes.toFixed(2)}`;
}

function scalarValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  return "—";
}
