import { HealthCheck } from "./HealthCheck";
// @ts-ignore
import brandLogo from "../assets/image.png";

export function Header() {
  return (
    <header className="p-6 mb-3 relative" style={{ border: 'none' }}>
      <div className="relative flex items-center justify-between">
        {/* Center - Title */}
        <div className="text-center flex-1 flex items-center justify-center gap-4">
          <img
            src={brandLogo}
            alt="逗点生物Logo"
            className="w-12 h-12 rounded-xl shadow-lg object-contain"
          />
          <h1>
            <span className="text-3xl font-semibold tracking-wider text-foreground" style={{ fontFamily: '"Space Grotesk", system-ui, sans-serif' }}>逗点生物食品分析助手</span>
          </h1>
        </div>

        {/* Right side - health check */}
        <div className="w-48 flex justify-end items-center">
          <HealthCheck />
        </div>
      </div>
    </header>
  );
}
