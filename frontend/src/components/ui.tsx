import {
  Alert,
  Badge,
  Box,
  Card,
  Center,
  Group,
  Loader,
  Paper,
  ScrollArea,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from "@mantine/core";
import { IconAlertCircle, IconChartBar, IconDatabase, IconInfoCircle } from "@tabler/icons-react";
import type { ReactNode } from "react";

export function SectionCard(props: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card withBorder radius="lg" shadow="sm" p="lg">
      <Group justify="space-between" align="flex-start" mb="md">
        <Stack gap={4}>
          <Title order={3}>{props.title}</Title>
          {props.description ? (
            <Text c="dimmed" size="sm">
              {props.description}
            </Text>
          ) : null}
        </Stack>
        {props.action}
      </Group>
      {props.children}
    </Card>
  );
}

export function EmptyState(props: { title: string; description: string; icon?: ReactNode }) {
  return (
    <Center py="xl">
      <Paper withBorder radius="lg" p="xl" w="100%" maw={760}>
        <Stack align="center" gap="md">
          <ThemeIcon variant="light" size={56} radius="xl">
            {props.icon ?? <IconDatabase size={28} />}
          </ThemeIcon>
          <Stack gap={6} align="center">
            <Title order={4}>{props.title}</Title>
            <Text c="dimmed" ta="center" maw={560}>
              {props.description}
            </Text>
          </Stack>
        </Stack>
      </Paper>
    </Center>
  );
}

export function LoadingState(props: { label?: string }) {
  return (
    <Center py="xl">
      <Stack align="center">
        <Loader size="lg" />
        <Text c="dimmed">{props.label ?? "Loading…"}</Text>
      </Stack>
    </Center>
  );
}

export function ErrorState(props: { title?: string; error: string }) {
  return (
    <Alert
      icon={<IconAlertCircle size={16} />}
      color="red"
      title={props.title ?? "Something went wrong"}
      variant="light"
    >
      {props.error}
    </Alert>
  );
}

export function MetricTiles(props: { items: Array<{ label: string; value: ReactNode; hint?: string }> }) {
  return (
    <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
      {props.items.map((item) => (
        <Paper key={item.label} withBorder radius="lg" p="md">
          <Stack gap={4}>
            <Text size="sm" c="dimmed">
              {item.label}
            </Text>
            <Title order={3}>{item.value}</Title>
            {item.hint ? (
              <Text size="xs" c="dimmed">
                {item.hint}
              </Text>
            ) : null}
          </Stack>
        </Paper>
      ))}
    </SimpleGrid>
  );
}

export function StatusBadge(props: { status: string }) {
  const color =
    props.status === "completed"
      ? "green"
      : props.status === "running"
        ? "blue"
        : props.status === "paused"
          ? "yellow"
          : props.status === "failed" || props.status === "cancelled"
            ? "red"
            : "gray";
  return <Badge color={color}>{props.status}</Badge>;
}

export function SubtleHint(props: { text: string }) {
  return (
    <Group gap="xs" align="flex-start" wrap="nowrap">
      <IconInfoCircle size={16} style={{ marginTop: 2, flexShrink: 0 }} />
      <Text size="sm" c="dimmed">
        {props.text}
      </Text>
    </Group>
  );
}

export function JsonScroller(props: { children: ReactNode }) {
  return (
    <ScrollArea h={280} type="auto">
      <Paper withBorder radius="md" p="sm" bg="gray.0">
        <Box style={{ whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: 12 }}>
          {props.children}
        </Box>
      </Paper>
    </ScrollArea>
  );
}

export function ChartEmpty(props: { title: string; description: string }) {
  return <EmptyState title={props.title} description={props.description} icon={<IconChartBar size={28} />} />;
}
