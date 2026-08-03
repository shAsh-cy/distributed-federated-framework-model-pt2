/**
 * Run configuration. Options come from the capabilities endpoints — nothing
 * here hardcodes a dataset, algorithm or architecture. Labels name what the
 * user controls ("Clients per round"), not internal field names.
 *
 * The alpha preview: dragging Dirichlet alpha rearranges the per-client
 * label histograms live, from near-uniform (alpha high) to pathological
 * single-class shards (alpha low) — the data-heterogeneity dial made visible
 * before any compute is spent.
 */
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api, ApiError } from "../lib/api";
import { previewPartition } from "../lib/dirichlet";
import { Button, Field, Select, Skeleton, Slider, TextInput, useToast } from "../ui/primitives";

const HISTOGRAM_COLOURS = ["var(--client)", "var(--global)"];

function MiniHistogram({ counts, max }: { counts: number[]; max: number }) {
  const width = 96;
  const barWidth = width / counts.length;
  return (
    <svg
      width={width}
      height={28}
      role="img"
      aria-label={`Label histogram: ${counts.join(", ")}`}
      className="shrink-0"
    >
      {counts.map((count, i) => {
        const h = max > 0 ? Math.max(count > 0 ? 1 : 0, (count / max) * 26) : 0;
        return (
          <rect
            key={i}
            x={i * barWidth + 0.5}
            y={28 - h}
            width={barWidth - 1}
            height={h}
            fill={HISTOGRAM_COLOURS[0]}
          />
        );
      })}
    </svg>
  );
}

export function ConfigureView({ onStarted }: { onStarted: (runId: string) => void }) {
  const toast = useToast();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const algorithms = useQuery({ queryKey: ["algorithms"], queryFn: api.algorithms });
  const architectures = useQuery({ queryKey: ["architectures"], queryFn: api.architectures });

  const [dataset, setDataset] = useState("fashion_mnist");
  const [numClients, setNumClients] = useState(10);
  const [clientsPerRound, setClientsPerRound] = useState(5);
  const [rounds, setRounds] = useState(20);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [alpha, setAlpha] = useState(0.5);
  const [algorithm, setAlgorithm] = useState("fedavg");
  const [clipNorm, setClipNorm] = useState(0.5);
  const [noiseMultiplier, setNoiseMultiplier] = useState(2.0);
  const [starting, setStarting] = useState(false);

  const activeDataset = datasets.data?.find((d) => d.name === dataset);
  const partitionScheme = activeDataset?.partition_schemes.includes("dirichlet")
    ? "dirichlet"
    : (activeDataset?.partition_schemes[0] ?? "dirichlet");
  const dpEnabled = algorithm === "dp-fedavg";
  const model = activeDataset?.model ?? "small_cnn";
  const architecture = architectures.data?.find((a) => a.name === model);

  const previewClients = Math.min(numClients, 12);
  const preview = useMemo(
    () =>
      partitionScheme === "dirichlet"
        ? previewPartition({
            alpha,
            numClients: previewClients,
            numClasses: activeDataset?.num_classes ?? 10,
            perClass: 600, // preview scale; shape is what matters
          })
        : null,
    [alpha, previewClients, activeDataset?.num_classes, partitionScheme],
  );
  const previewMax = preview ? Math.max(...preview.flat()) : 0;

  async function start() {
    setStarting(true);
    try {
      const config = {
        seed: 42,
        data: {
          dataset,
          num_clients: numClients,
          partition: partitionScheme,
          ...(partitionScheme === "dirichlet" ? { dirichlet_alpha: alpha } : {}),
        },
        model: { name: model },
        training: {
          rounds,
          client_fraction: clientsPerRound / numClients,
          local_epochs: localEpochs,
        },
        server: { min_clients_per_round: Math.min(2, clientsPerRound) },
        privacy: dpEnabled
          ? { enabled: true, l2_clip_norm: clipNorm, noise_multiplier: noiseMultiplier }
          : { enabled: false },
      };
      const { run_id } = await api.startRun(config);
      onStarted(run_id);
    } catch (error) {
      const message =
        error instanceof ApiError
          ? `The server rejected this configuration: ${error.detail}. Adjust the highlighted values and try again.`
          : "Could not reach the coordinator. Check that the API is running, then retry.";
      toast(message, "error");
    } finally {
      setStarting(false);
    }
  }

  if (datasets.isLoading || algorithms.isLoading || architectures.isLoading) {
    return <Skeleton lines={6} label="Capabilities" />;
  }
  if (datasets.isError || !datasets.data) {
    return (
      <p className="font-prose text-base">
        The capabilities endpoints did not answer, so there is nothing to configure yet. Start
        the coordinator API and reload.
      </p>
    );
  }

  return (
    <div className="grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <form
        className="flex flex-col gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          void start();
        }}
      >
        <Field label="Dataset">
          {(id) => (
            <Select id={id} value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {datasets.data.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.name} · {d.num_classes} classes
                </option>
              ))}
            </Select>
          )}
        </Field>

        <div className="readout border border-rule bg-ground-raised px-3 py-2 text-xs">
          architecture {model} · {architecture?.parameter_count.toLocaleString("en-US")}{" "}
          parameters
        </div>

        <Field label="Client population">
          {(id) => (
            <TextInput
              id={id}
              type="number"
              min={2}
              max={3400}
              value={numClients}
              onChange={(e) => setNumClients(Number(e.target.value))}
            />
          )}
        </Field>

        <Slider
          label="Clients per round"
          value={clientsPerRound}
          min={1}
          max={numClients}
          step={1}
          onChange={setClientsPerRound}
          format={(v) => `${v} of ${numClients}`}
        />
        <Slider label="Rounds" value={rounds} min={1} max={200} step={1} onChange={setRounds} />
        <Slider
          label="Local epochs"
          value={localEpochs}
          min={1}
          max={10}
          step={1}
          onChange={setLocalEpochs}
        />

        <Field label="Algorithm">
          {(id) => (
            <Select id={id} value={algorithm} onChange={(e) => setAlgorithm(e.target.value)}>
              {algorithms.data?.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.name}
                </option>
              ))}
            </Select>
          )}
        </Field>

        {dpEnabled ? (
          <>
            <Slider
              label="Clipping norm"
              value={clipNorm}
              min={0.0625}
              max={3}
              step={0.0625}
              onChange={setClipNorm}
              format={(v) => v.toFixed(4)}
            />
            <Slider
              label="Noise multiplier"
              value={noiseMultiplier}
              min={0.1}
              max={6}
              step={0.1}
              onChange={setNoiseMultiplier}
              format={(v) => v.toFixed(1)}
            />
            <p className="font-prose text-xs text-slate">
              Epsilon is computed by the accountant from noise, sampling rate and rounds — it
              is never set directly. The console meter shows it accumulate.
            </p>
          </>
        ) : null}

        <Button tone="primary" type="submit" disabled={starting}>
          {starting ? "Starting run" : "Start run"}
        </Button>
      </form>

      <section aria-label="Partition preview" className="flex flex-col gap-3">
        {partitionScheme === "dirichlet" && preview ? (
          <>
            <Slider
              label="Dirichlet alpha"
              value={alpha}
              min={0.05}
              max={10}
              step={0.05}
              onChange={setAlpha}
              format={(v) => v.toFixed(2)}
            />
            <p className="font-prose text-xs text-slate">
              Per-client label distributions this alpha produces
              {numClients > previewClients ? ` (first ${previewClients} clients shown)` : ""}.
              Statistical preview of the split's shape; the run draws its own seeded
              partition.
            </p>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
              {preview.map((counts, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="readout w-8 text-xs text-slate">c{i}</span>
                  <MiniHistogram counts={counts} max={previewMax} />
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="font-prose text-sm">
            {dataset} is naturally partitioned — each client is one real writer, so there is
            no synthetic split to tune. The population control selects how many writers take
            part.
          </p>
        )}
      </section>
    </div>
  );
}
