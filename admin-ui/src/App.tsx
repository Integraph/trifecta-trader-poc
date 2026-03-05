import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { DashboardPage }  from './components/dashboard/DashboardPage';
import { SchedulerPage }  from './components/scheduler/SchedulerPage';
import { QueuePage }      from './components/queue/QueuePage';
import { AccuracyPage }   from './components/accuracy/AccuracyPage';
import { TestRunPage }    from './components/test-run/TestRunPage';
import { AnalysesPage }   from './components/analyses/AnalysesPage';
import { ConfigPage }     from './components/config/ConfigPage';
import { LogsPage }       from './components/logs/LogsPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index             element={<DashboardPage />} />
          <Route path="scheduler"  element={<SchedulerPage />} />
          <Route path="queue"      element={<QueuePage />} />
          <Route path="accuracy"   element={<AccuracyPage />} />
          <Route path="test-run"   element={<TestRunPage />} />
          <Route path="analyses"   element={<AnalysesPage />} />
          <Route path="config"     element={<ConfigPage />} />
          <Route path="logs"       element={<LogsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
