/**
 * The closing card: what this is, and where to go next.
 *
 * docs/how-it-works.html is the same six stages as a single self-contained
 * page that opens on a double-click — the version you can send to someone who
 * will never run a server.
 */
import { Fig } from "../ui";

const REPO = "https://github.com/shAsh-cy/distributed-federated-framework-model-pt2";

export function ClosingCard() {
  return (
    <div className="flex flex-col gap-4">
      <p className="story-measure font-prose text-base">
        This project trains a shared model across clients that never send their data anywhere,
        measures what the privacy mechanism costs instead of asserting that it is free, and
        publishes the runs behind every number — including the ones that went badly.
      </p>
      <p className="story-measure font-prose text-base">
        The headline: on FEMNIST with <Fig name="cohort" /> clients per round for{" "}
        <Fig name="rounds" /> rounds, client-level differential privacy at ε <Fig name="epsilon" />{" "}
        cost <Fig name="dpCost" /> points of accuracy. On the easier Fashion-MNIST benchmark the
        same stack reaches <Fig name="fashionDense" /> without privacy.
      </p>
      <ul className="flex flex-col gap-2 font-prose text-base">
        <li>
          <a className="underline decoration-rule underline-offset-4 hover:text-global" href={REPO}>
            The repository
          </a>{" "}
          — code, configs, and every results JSON quoted in this walkthrough.
        </li>
        <li>
          <a
            className="underline decoration-rule underline-offset-4 hover:text-global"
            href={REPO + "/blob/main/docs/how-it-works.html"}
          >
            docs/how-it-works.html
          </a>{" "}
          — this same story as one file you can save and email to someone. No server, no build
          step; it opens on a double-click.
        </li>
      </ul>
    </div>
  );
}
