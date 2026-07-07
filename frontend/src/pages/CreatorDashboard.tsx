import { useState } from "react";
import { Badge, Button, Card, FileInput, Group, SimpleGrid, Stack, Text, TextInput } from "@mantine/core";
import { IconPlayerPlay } from "@tabler/icons-react";

import { api, downloadBlob } from "@/api/client";
import type { JobSnapshot, PredictionHorizonResponse, PredictionResult } from "@/api/types";
import { ChartCard, SimpleBarChart, SimpleComposedChart } from "@/components/charts";
import { EmptyState, ErrorState, LoadingState, MetricTiles, SectionCard, StatusBadge, SubtleHint } from "@/components/ui";
import { useJobSnapshot } from "@/hooks/useJobSnapshot";
import { usePolling } from "@/hooks/usePolling";
import { formatNumber, formatPercent } from "@/utils/format";

export function CreatorDashboard() {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const horizon = usePolling<PredictionHorizonResponse>(() => api.prediction.horizon(), [], 60000);
  const snapshot = useJobSnapshot<JobSnapshot>(jobId, api.prediction.job, 2000);
  const result = snapshot.data?.result as PredictionResult | undefined;
  const explanation = isRecord(result?.explanation) ? result.explanation : null;

  async function submitUrl() {
    if (!url.trim()) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const response = await api.prediction.url({ url: url.trim(), horizon_days: horizon.data?.horizon_days ?? null });
      setJobId(response.job_id);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Prediction failed");
      setSubmitting(false);
    }
  }

  async function submitFile() {
    if (!file) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const response = await api.prediction.upload(file);
      setJobId(response.job_id);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Upload failed");
      setSubmitting(false);
    }
  }

  function downloadResult() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    downloadBlob(blob, "prediction-result.json");
  }

  return (
    <Stack gap="lg">
      <SectionCard
        title="Creator prediction"
        description="Submit a YouTube URL or upload a local video file. The backend handles ingest, TRIBE inference, and explainability."
      >
        <Stack gap="md">
          <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
            <Card withBorder radius="lg" p="md">
              <Stack gap="md">
                <TextInput label="YouTube URL" placeholder="https://www.youtube.com/watch?v=..." value={url} onChange={(event) => setUrl(event.currentTarget.value)} />
                <Group>
                  <Button leftSection={<IconPlayerPlay size={16} />} onClick={() => void submitUrl()}>
                    Predict from URL
                  </Button>
                  <Badge variant="light">Job-based flow</Badge>
                </Group>
              </Stack>
            </Card>

            <Card withBorder radius="lg" p="md">
              <Stack gap="md">
                <FileInput label="Local video file" value={file} onChange={setFile} placeholder="Upload a video file" />
                <Group>
                  <Button variant="light" onClick={() => void submitFile()}>
                    Predict from file
                  </Button>
                  <Badge variant="light">{horizon.data ? `${horizon.data.horizon_days} day horizon` : "Loading horizon…"}</Badge>
                </Group>
              </Stack>
            </Card>
          </SimpleGrid>

          {submitError ? <ErrorState error={submitError} /> : null}
          {submitting ? <LoadingState label="TRIBE inference is running on CPU. This may take a while…" /> : null}

          {snapshot.data ? (
            <Card withBorder radius="lg" p="md">
              <Group justify="space-between">
                <Stack gap={2}>
                  <Text fw={600}>Prediction job</Text>
                  <Text size="sm" c="dimmed">
                    {jobId}
                  </Text>
                </Stack>
                <StatusBadge status={snapshot.data.status} />
              </Group>
            </Card>
          ) : (
            <EmptyState title="Ready for a prediction" description="Enter a URL or upload a file. The dashboard stays clean until the backend returns a real result." />
          )}
        </Stack>
      </SectionCard>

      {result ? (
        <Stack gap="md">
          <MetricTiles
            items={[
              { label: "Predicted likes", value: formatNumber(result.prediction.likes ?? 0, 0), hint: result.horizon_label },
              { label: "Predicted views", value: formatNumber(result.prediction.views ?? 0, 0) },
              { label: "Engagement", value: formatPercent(result.prediction.engagement) },
              { label: "Virality", value: formatPercent(result.prediction.virality) },
            ]}
          />

          <SectionCard
            title="Prediction result"
            description={result.horizon_label}
            action={<Button variant="light" onClick={downloadResult}>Download JSON</Button>}
          >
            <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md">
              <ChartCard title="Outcome summary" height={280}>
                <SimpleBarChart
                  data={[
                    {
                      label: "Prediction",
                      likes: result.prediction.likes ?? 0,
                      views: result.prediction.views ?? 0,
                      engagement: result.prediction.engagement ?? 0,
                      virality: result.prediction.virality ?? 0,
                    },
                  ]}
                  xKey="label"
                  bars={["likes", "views", "engagement", "virality"]}
                />
              </ChartCard>

              <ChartCard title="Confidence intervals" height={280}>
                <SimpleComposedChart data={confidenceRows(explanation)} xKey="label" bars={["low", "high"]} lines={["point"]} />
              </ChartCard>
            </SimpleGrid>

            <SimpleGrid cols={{ base: 1, xl: 2 }} spacing="md" mt="md">
              <Card withBorder radius="lg" p="md">
                <Text fw={600} mb="sm">
                  Explainability summary
                </Text>
                <Stack gap="xs">
                  <SummaryLine label="Brain activation" value={summaryValue(explanation, ["brain_activation", "activation_summary", "attention"])} />
                  <SummaryLine label="Feature importance" value={summaryValue(explanation, ["feature_importance", "brain_region_importance"])} />
                  <SummaryLine label="Temporal segments" value={summaryValue(explanation, ["temporal_segments", "attention_timeline"])} />
                  <SummaryLine label="Confidence interval" value={summaryValue(explanation, ["confidence_interval", "intervals"])} />
                </Stack>
              </Card>

              <Card withBorder radius="lg" p="md">
                <Text fw={600} mb="sm">
                  Top influential temporal segments
                </Text>
                {segmentRows(explanation).length > 0 ? (
                  <ChartCard title="Temporal influence" height={240}>
                    <SimpleBarChart data={segmentRows(explanation)} xKey="segment" bars={["value"]} />
                  </ChartCard>
                ) : (
                  <EmptyState title="No temporal segments exposed" description="When the backend returns segment-level explainability, it will appear here." />
                )}
              </Card>
            </SimpleGrid>
          </SectionCard>
        </Stack>
      ) : (
        <SectionCard title="No prediction yet" description="The prediction result stays empty until the job completes.">
          <SubtleHint text="This view never invents values — if the backend is empty, it stays a polished empty state." />
        </SectionCard>
      )}
    </Stack>
  );
}

function SummaryLine(props: { label: string; value: string }) {
  return (
    <Group justify="space-between" align="flex-start">
      <Text size="sm" c="dimmed">
        {props.label}
      </Text>
      <Text size="sm" ta="right" maw={320}>
        {props.value}
      </Text>
    </Group>
  );
}

function summaryValue(explanation: Record<string, unknown> | null, keys: string[]): string {
  if (!explanation) return "—";
  for (const key of keys) {
    const value = explanation[key];
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return `${value.length} items`;
    if (isRecord(value)) return `${Object.keys(value).length} fields`;
  }
  return "—";
}

function confidenceRows(explanation: Record<string, unknown> | null): Array<Record<string, number | string>> {
  if (!explanation) return [{ label: "Prediction", low: 0, high: 0, point: 0 }];
  const intervals = explanation.intervals;
  if (!isRecord(intervals)) return [{ label: "Prediction", low: 0, high: 0, point: 0 }];
  return Object.entries(intervals).map(([label, value]) => {
    if (!isRecord(value)) {
      return { label, low: 0, high: 0, point: 0 };
    }
    return {
      label,
      low: numberValue(value.lower),
      high: numberValue(value.upper),
      point: numberValue(value.point),
    };
  });
}

function segmentRows(explanation: Record<string, unknown> | null): Array<Record<string, number | string>> {
  if (!explanation) return [];
  const segments = explanation.temporal_segments ?? explanation.attention_timeline;
  if (!Array.isArray(segments)) return [];
  return segments.slice(0, 12).map((segment, index) => {
    if (!isRecord(segment)) {
      return { segment: String(index + 1), value: numberValue(segment) };
    }
    const label = segment.label ?? segment.name ?? index + 1;
    return {
      segment: typeof label === "string" || typeof label === "number" ? String(label) : String(index + 1),
      value: numberValue(segment.value ?? segment.score ?? segment.importance),
    };
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
