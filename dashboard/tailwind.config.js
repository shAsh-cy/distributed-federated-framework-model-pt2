/** Instrument palette and type scale — the only sizes and colours that exist.
 * Everything here mirrors src/styles/tokens.css; Tailwind utilities are the
 * delivery mechanism, tokens.css is the source of truth for raw values.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: ["selector", '[data-theme="dark"]'],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      ground: {
        DEFAULT: "var(--ground)",
        raised: "var(--ground-raised)",
      },
      ink: "var(--ink)",
      global: "var(--global)", // Prussian blue: the global model
      client: "var(--client)", // oxide red: client-local signals
      budget: "var(--budget)", // ochre: privacy budget ONLY
      slate: "var(--slate)", // disabled / dropped / inactive
      rule: "var(--rule)",
    },
    fontFamily: {
      head: ["Bahnschrift", "Arial Narrow", "Roboto Condensed", "sans-serif"],
      mono: [
        "Cascadia Mono",
        "JetBrains Mono",
        "Consolas",
        "ui-monospace",
        "monospace",
      ],
      prose: ["system-ui", "Segoe UI", "sans-serif"],
    },
    fontSize: {
      // Explicit scale. No arbitrary sizes anywhere else.
      xs: ["0.6875rem", { lineHeight: "1rem" }], // 11px small readouts
      sm: ["0.8125rem", { lineHeight: "1.25rem" }], // 13px body mono
      base: ["0.9375rem", { lineHeight: "1.5rem" }], // 15px prose
      lg: ["1.125rem", { lineHeight: "1.5rem" }], // 18px section heads
      xl: ["1.5rem", { lineHeight: "1.75rem" }], // 24px view titles
      "2xl": ["2.25rem", { lineHeight: "2.5rem" }], // 36px hero numerals
    },
    extend: {
      letterSpacing: { head: "0.02em" },
    },
  },
  plugins: [],
};
