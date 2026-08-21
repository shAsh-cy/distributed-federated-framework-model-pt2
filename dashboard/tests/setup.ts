import "@testing-library/jest-dom/vitest";
import { afterAll, afterEach, beforeAll } from "vitest";

import { server } from "../src/mocks/node";

beforeAll(() => server.listen({ onUnhandledRequest: "bypass" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// jsdom lacks ResizeObserver (recharts ResponsiveContainer needs it).
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!("ResizeObserver" in globalThis)) {
  (globalThis as Record<string, unknown>).ResizeObserver = ResizeObserverStub;
}

// jsdom has no matchMedia. The topology asks it for the viewport width and
// framer-motion asks it for prefers-reduced-motion, so without this every
// component that renders a topology throws on mount. Answers "no" to
// everything: the wide layout, motion on. Tests that need the other answers
// use Playwright, where the media state is real.
class MediaQueryListStub implements Partial<MediaQueryList> {
  constructor(public media: string) {}
  matches = false;
  onchange = null;
  addEventListener() {}
  removeEventListener() {}
  addListener() {}
  removeListener() {}
  dispatchEvent() {
    return false;
  }
}
if (typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => new MediaQueryListStub(query) as unknown as MediaQueryList,
  });
}
