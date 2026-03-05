import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { useHealth } from '../../api/hooks';

export function Layout() {
  const { data: health } = useHealth(10_000);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar health={health} />
      <div className="flex flex-col flex-1 min-w-0">
        <Header health={health} />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="max-w-[1400px] mx-auto">
            <Outlet context={{ health }} />
          </div>
        </main>
      </div>
    </div>
  );
}
