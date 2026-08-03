/**
 * The eight primitives. Everything else in the application is bespoke and
 * composed from these plus raw SVG. Instrument styling throughout: hairline
 * rules, flat surfaces, monospace readouts, no shadows-as-decoration.
 */
import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

/* -- Button --------------------------------------------------------------- */

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: "default" | "primary" | "danger";
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { tone = "default", className = "", ...rest },
  ref,
) {
  const tones: Record<string, string> = {
    default: "border-ink text-ink hover:bg-ground-raised",
    primary: "border-global bg-global text-ground-raised hover:opacity-90",
    danger: "border-client text-client hover:bg-ground-raised",
  };
  return (
    <button
      ref={ref}
      className={`inline-flex items-center gap-2 border px-3 py-1.5 font-head text-sm tracking-head uppercase disabled:border-slate disabled:text-slate disabled:cursor-not-allowed ${tones[tone]} ${className}`}
      {...rest}
    />
  );
});

/* -- Field ---------------------------------------------------------------- */

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: (id: string) => ReactNode;
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={id} className="font-head text-xs uppercase tracking-head text-ink">
        {label}
      </label>
      {children(id)}
      {hint ? <p className="font-prose text-xs text-slate">{hint}</p> : null}
    </div>
  );
}

/* -- TextInput / NumberInput ---------------------------------------------- */

export const TextInput = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function TextInput({ className = "", ...rest }, ref) {
    return (
      <input
        ref={ref}
        className={`readout border border-rule bg-ground-raised px-2 py-1.5 text-sm text-ink focus:border-global ${className}`}
        {...rest}
      />
    );
  },
);

/* -- Select --------------------------------------------------------------- */

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  function Select({ className = "", children, ...rest }, ref) {
    return (
      <select
        ref={ref}
        className={`readout border border-rule bg-ground-raised px-2 py-1.5 text-sm text-ink focus:border-global ${className}`}
        {...rest}
      >
        {children}
      </select>
    );
  },
);

/* -- Slider (labelled range with live mono readout) ------------------------ */

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => String(v),
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="font-head text-xs uppercase tracking-head">
          {label}
        </label>
        <output htmlFor={id} className="readout text-sm">
          {format(value)}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-[var(--global)]"
      />
    </div>
  );
}

/* -- Meter (the ochre one is the privacy budget; tone must be passed) ------ */

export function Meter({
  label,
  value,
  max,
  tone,
  format,
}: {
  label: string;
  value: number;
  max: number;
  tone: "budget" | "global";
  format: (v: number, max: number) => string;
}) {
  const fraction = max > 0 ? Math.min(1, value / max) : 0;
  const colour = tone === "budget" ? "var(--budget)" : "var(--global)";
  return (
    <div aria-label={`${label}: ${format(value, max)}`} className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="font-head text-xs uppercase tracking-head">{label}</span>
        <span className="readout text-sm">{format(value, max)}</span>
      </div>
      <div className="h-2 border border-rule bg-ground-raised" role="presentation">
        <div
          className="h-full transition-[width]"
          style={{ width: `${fraction * 100}%`, background: colour }}
        />
      </div>
    </div>
  );
}

/* -- Dialog ---------------------------------------------------------------- */

export function Dialog({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onCancel={onClose}
      className="border border-ink bg-ground p-0 text-ink backdrop:bg-ink/40"
    >
      <div className="flex items-center justify-between border-b border-rule px-4 py-2">
        <h2 className="font-head text-lg uppercase tracking-head">{title}</h2>
        <Button onClick={onClose} aria-label="Close dialog">
          Close
        </Button>
      </div>
      <div className="p-4">{children}</div>
    </dialog>
  );
}

/* -- Toast ----------------------------------------------------------------- */

type ToastMessage = { id: number; text: string; tone: "info" | "error" };
const ToastContext = createContext<(text: string, tone?: "info" | "error") => void>(() => {});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const push = useCallback((text: string, tone: "info" | "error" = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, tone }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 6000);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div aria-live="polite" className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`border px-3 py-2 font-prose text-sm bg-ground-raised ${
              t.tone === "error" ? "border-client text-client" : "border-ink text-ink"
            }`}
          >
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

/* -- Skeleton (loading is skeletal, never a spinner) ----------------------- */

export function Skeleton({ lines = 3, label }: { lines?: number; label: string }) {
  return (
    <div role="status" aria-label={`${label} loading`} className="flex flex-col gap-2">
      {Array.from({ length: lines }, (_, i) => (
        <div
          key={i}
          className="h-3 animate-pulse bg-rule"
          style={{ width: `${88 - i * 13}%` }}
        />
      ))}
    </div>
  );
}
