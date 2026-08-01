/** Stub — the animated topology lands in its own commit. */
import type { RunState } from "../lib/events";

export function Topology(_props: {
  run: RunState;
  highlightRound: number | null;
  onPin: (clientId: string | null) => void;
  pinned: string | null;
}) {
  return (
    <p className="font-prose text-sm text-slate">
      Topology arrives in the next commit; the console below is already live.
    </p>
  );
}
