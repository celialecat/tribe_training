import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Paper, Text } from "@mantine/core";
import type { ReactNode } from "react";

export function ChartCard(props: {
  title: string;
  children: ReactNode;
  height?: number;
  description?: string;
}) {
  return (
    <Paper withBorder radius="lg" p="md">
      <Text fw={600} mb={props.description ? 4 : "sm"}>
        {props.title}
      </Text>
      {props.description ? (
        <Text size="sm" c="dimmed" mb="sm">
          {props.description}
        </Text>
      ) : null}
      <div style={{ width: "100%", height: props.height ?? 280 }}>{props.children}</div>
    </Paper>
  );
}

export function SimpleLineChart(props: { data: Array<Record<string, unknown>>; xKey: string; lines: string[] }) {
  return (
    <ResponsiveContainer>
      <LineChart data={props.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={props.xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {props.lines.map((line, index) => (
          <Line
            key={line}
            type="monotone"
            dataKey={line}
            stroke={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SimpleAreaChart(props: { data: Array<Record<string, unknown>>; xKey: string; areas: string[] }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={props.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={props.xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {props.areas.map((area, index) => (
          <Area
            key={area}
            type="monotone"
            dataKey={area}
            fill={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]}
            stroke={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]}
            fillOpacity={0.2}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function SimpleBarChart(props: {
  data: Array<Record<string, unknown>>;
  xKey: string;
  bars: string[];
}) {
  return (
    <ResponsiveContainer>
      <BarChart data={props.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={props.xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {props.bars.map((bar, index) => (
          <Bar key={bar} dataKey={bar} radius={[6, 6, 0, 0]} fill={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]}>
            {props.data.map((_, dataIndex) => (
              <Cell key={`${bar}-${dataIndex}`} fill={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]} />
            ))}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SimpleComposedChart(props: {
  data: Array<Record<string, unknown>>;
  xKey: string;
  bars: string[];
  lines: string[];
}) {
  return (
    <ResponsiveContainer>
      <ComposedChart data={props.data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey={props.xKey} />
        <YAxis />
        <Tooltip />
        <Legend />
        {props.bars.map((bar, index) => (
          <Bar
            key={bar}
            dataKey={bar}
            radius={[6, 6, 0, 0]}
            fill={["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b"][index % 4]}
            opacity={0.4}
          />
        ))}
        {props.lines.map((line, index) => (
          <Line
            key={line}
            type="monotone"
            dataKey={line}
            stroke={["#111827", "#4f46e5", "#0ea5e9", "#14b8a6"][index % 4]}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
