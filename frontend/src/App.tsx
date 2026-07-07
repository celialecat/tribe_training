import { useMemo } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import {
  AppShell,
  Badge,
  Burger,
  Group,
  NavLink as MantineNavLink,
  ScrollArea,
  Stack,
  Text,
  Title,
  useMantineTheme,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { IconBrain, IconPlayerPlay, IconSparkles } from "@tabler/icons-react";

import { CreatorDashboard } from "./pages/CreatorDashboard";
import { ScientificDashboard } from "./pages/ScientificDashboard";

export default function App() {
  const [opened, { toggle }] = useDisclosure(false);
  const location = useLocation();
  const theme = useMantineTheme();
  const items = useMemo(
    () => [
      {
        to: "/scientific",
        label: "Scientific dashboard",
        icon: <IconBrain size={18} />,
        description: "Research workflow and latent analysis",
      },
      {
        to: "/creator",
        label: "Creator dashboard",
        icon: <IconPlayerPlay size={18} />,
        description: "Prediction and explainability",
      },
    ],
    [],
  );

  return (
    <AppShell
      header={{ height: 72 }}
      navbar={{ width: 280, breakpoint: "sm", collapsed: { mobile: !opened } }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="lg" justify="space-between">
          <Group>
            <Burger opened={opened} onClick={toggle} hiddenFrom="sm" size="sm" />
            <Stack gap={0}>
              <Title order={3}>TRIBE Dashboards</Title>
              <Text size="sm" c="dimmed">
                Clean, live, backend-driven workflows for researchers and creators.
              </Text>
            </Stack>
          </Group>
          <Group gap="xs">
            <Badge variant="light" color="indigo">
              Local backend
            </Badge>
            <Badge variant="light" color="teal">
              Empty DB safe
            </Badge>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="md">
        <AppShell.Section grow>
          <ScrollArea h="100%">
          <Stack gap="xs">
            {items.map((item) => (
              <MantineNavLink
                key={item.to}
                component={NavLink}
                to={item.to}
                label={item.label}
                description={item.description}
                leftSection={item.icon}
                active={location.pathname === item.to}
                variant="light"
              />
            ))}
          </Stack>
          </ScrollArea>
        </AppShell.Section>
        <AppShell.Section pt="md">
          <Text size="xs" c="dimmed">
            <IconSparkles size={12} style={{ marginRight: 4 }} />
            Built with Mantine, Recharts, and live backend endpoints.
          </Text>
        </AppShell.Section>
      </AppShell.Navbar>

      <AppShell.Main
        style={{
          background: theme.white,
          borderRadius: theme.radius.lg,
          minHeight: "calc(100vh - 104px)",
        }}
      >
        <Routes>
          <Route path="/" element={<Navigate to="/scientific" replace />} />
          <Route path="/scientific" element={<ScientificDashboard />} />
          <Route path="/creator" element={<CreatorDashboard />} />
        </Routes>
      </AppShell.Main>
    </AppShell>
  );
}
