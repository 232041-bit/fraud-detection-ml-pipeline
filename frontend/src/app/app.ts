import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { FEATURES, Feature, SAMPLE_FRAUD, SAMPLE_LEGIT, SAMPLE_REVIEW } from './feature-schema';
import {
  DriftBatchResponse, DriftLogEntry, PolicyState, PredictionResponse, PredictionService,
} from './prediction.service';

type Tab = 'predict' | 'drift' | 'research';

const POLICY_COLORS: Record<string, string> = {
  static: '#8a94a6', drift: '#2a6fdb', drift_fair: '#16a37f', always: '#e07b24',
};
const MODEL_COLORS: Record<string, string> = {
  random_forest: '#2a6fdb', deep_ensemble: '#16a37f', mc_dropout: '#c2410c',
};

@Component({
  selector: 'app-root',
  imports: [FormsModule, DecimalPipe, PercentPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields: Feature[] = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);
  readonly tab = signal<Tab>('predict');

  readonly models = [
    { id: 'random_forest', label: 'Random Forest (tree-disagreement uncertainty)' },
    { id: 'deep_ensemble', label: 'Deep Ensemble – 5 MLPs (Novelty 2)' },
    { id: 'mc_dropout', label: 'MC-Dropout MLP (Novelty 2)' },
    { id: 'logistic_regression', label: 'Logistic Regression (entropy only)' },
  ];
  selectedModel = 'random_forest';
  fairMode = false;

  values: Record<string, string | number | null> = {};
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);

  // ---- drift monitor (Novelty 1) ----
  driftModel = 'random_forest';
  driftMonth = 7;
  driftLabels = true;
  driftFair = false;
  readonly driftBusy = signal(false);
  readonly driftError = signal('');
  readonly driftLast = signal<DriftBatchResponse | null>(null);
  readonly driftLog = signal<DriftLogEntry[]>([]);
  readonly driftPolicy = signal<PolicyState | null>(null);

  // ---- research results ----
  readonly summary = signal<any>(null);
  readonly policyColors = POLICY_COLORS;
  readonly modelColors = MODEL_COLORS;
  readonly policyNames = ['static', 'drift', 'drift_fair', 'always'];
  readonly policyLabels: Record<string, string> = {
    static: 'Static threshold (baseline)', drift: 'Drift-aware (Novelty 1)',
    drift_fair: 'Drift-aware + fair', always: 'Recalibrate every batch (ablation)',
  };
  readonly modelLabels: Record<string, string> = {
    random_forest: 'Random Forest', deep_ensemble: 'Deep Ensemble', mc_dropout: 'MC Dropout',
    logistic_regression: 'Logistic Regression',
  };

  private api = inject(PredictionService);

  constructor() {
    for (const field of this.fields) this.values[field.name] = null;
    this.api.summary().subscribe({ next: (s) => this.summary.set(s), error: () => {} });
  }

  // =========================================================== prediction
  loadSample(kind: 'legit' | 'fraud' | 'review') {
    const sample = kind === 'legit' ? SAMPLE_LEGIT : kind === 'fraud' ? SAMPLE_FRAUD : SAMPLE_REVIEW;
    for (const field of this.fields) this.values[field.name] = sample[field.name];
    this.error.set('');
    this.result.set(null);
  }

  clearForm() {
    for (const field of this.fields) this.values[field.name] = null;
    this.error.set('');
    this.result.set(null);
  }

  percent(p?: number | null, digits = 2): string {
    return p === undefined || p === null ? '–' : (p * 100).toFixed(digits) + '%';
  }

  decisionText(d?: string): string {
    return d === 'FLAG_FRAUD' ? 'Flag as fraud' : d === 'MANUAL_REVIEW' ? 'Send to manual review'
      : d === 'APPROVE' ? 'Approve' : '';
  }

  /** position (0-100%) of a probability on the risk bar */
  pos(p?: number): number {
    return Math.max(0, Math.min(100, (p ?? 0) * 100));
  }

  submit() {
    this.error.set('');
    this.result.set(null);
    if (this.deployedStatic) {
      this.error.set('Static GitHub Pages demo: start Angular + Flask locally for predictions.');
      return;
    }
    const payload: Record<string, string | number> = {};
    for (const field of this.fields) {
      const value = this.values[field.name];
      if (value === null || value === '') {
        this.error.set(`Enter a value for ${field.label}`);
        return;
      }
      if (field.kind === 'number') {
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) {
          this.error.set(`Invalid number for ${field.label}`);
          return;
        }
        if ((field.min !== undefined && numeric < field.min) ||
            (field.max !== undefined && numeric > field.max)) {
          this.error.set(`Out-of-range value for ${field.label} (${field.min} to ${field.max})`);
          return;
        }
        payload[field.name] = numeric;
      } else {
        payload[field.name] = String(value);
      }
    }
    this.loading.set(true);
    this.api.predict(payload, this.selectedModel, this.fairMode).subscribe({
      next: (res) => { this.result.set(res); this.loading.set(false); },
      error: (err) => {
        this.error.set(`API error ${err.status || ''}: ` +
          (err.error?.error || err.message || 'Check Flask terminal'));
        this.loading.set(false);
      },
    });
  }

  // =========================================================== drift monitor
  sendDriftBatch() {
    if (this.deployedStatic) {
      this.driftError.set('The live drift monitor needs the local Flask API.');
      return;
    }
    this.driftBusy.set(true);
    this.driftError.set('');
    this.api.demoBatch(this.driftMonth).subscribe({
      next: (batch) => this.api.sendBatch(this.driftModel, batch, this.driftLabels, this.driftFair)
        .subscribe({
          next: (res) => {
            this.driftLast.set(res);
            this.driftPolicy.set(res.policy);
            this.driftLog.update((log) => [...log, res.update]);
            this.driftBusy.set(false);
            if (this.driftMonth < 7) this.driftMonth++;
          },
          error: (err) => this.failDrift(err),
        }),
      error: (err) => this.failDrift(err),
    });
  }

  resetDrift() {
    if (this.deployedStatic) return;
    this.api.resetDrift(this.driftModel, this.driftFair).subscribe({
      next: (res) => {
        this.driftPolicy.set(res.policy);
        this.driftLog.set([]);
        this.driftLast.set(null);
        this.driftMonth = 0;
      },
      error: (err) => this.failDrift(err),
    });
  }

  onDriftModelChange() {
    this.driftLog.set([]);
    this.driftLast.set(null);
    if (this.deployedStatic) return;
    this.api.driftStatus(this.driftModel).subscribe({
      next: (s) => { this.driftPolicy.set(s.policy); this.driftLog.set(s.log); },
      error: (err) => this.failDrift(err),
    });
  }

  private failDrift(err: any) {
    this.driftError.set(`API error ${err.status || ''}: ` +
      (err.error?.error || err.message || 'Check Flask terminal'));
    this.driftBusy.set(false);
  }

  readonly thresholdSeries = computed(() => {
    const log = this.driftLog();
    if (!log.length) return null;
    const pts = [log[0].threshold_before, ...log.map((e) => e.threshold_after)];
    return this.linePath(pts, 560, 140, Math.min(...pts) * 0.95, Math.max(...pts) * 1.05);
  });

  // =========================================================== charts (inline SVG)
  linePath(values: number[], w: number, h: number, lo?: number, hi?: number, pad = 8): string {
    if (!values?.length) return '';
    const min = lo ?? Math.min(...values);
    const max = hi ?? Math.max(...values);
    const span = max - min || 1;
    const step = values.length > 1 ? (w - 2 * pad) / (values.length - 1) : 0;
    return values.map((v, i) =>
      `${i ? 'L' : 'M'}${(pad + i * step).toFixed(1)},${(h - pad - ((v - min) / span) * (h - 2 * pad)).toFixed(1)}`,
    ).join(' ');
  }

  y(v: number, h: number, lo: number, hi: number, pad = 8): number {
    return h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  }

  x(i: number, n: number, w: number, pad = 8): number {
    return n > 1 ? pad + i * ((w - 2 * pad) / (n - 1)) : pad;
  }

  range(n: number): number[] {
    return Array.from({ length: n }, (_, i) => i);
  }

  heat(v: number): string {
    // PSI colour scale: stable (<0.1) -> moderate -> major (>=0.25)
    const t = Math.min(1, v / 0.5);
    const hue = 140 - 140 * t;
    return `hsl(${hue}, 65%, ${88 - 38 * t}%)`;
  }

  maxOf(arr: number[]): number { return Math.max(...arr); }

  keys(o: any): string[] { return o ? Object.keys(o) : []; }

  sweepPath(sweep: any[], w: number, h: number, xmax: number, ymax: number): string {
    const pad = 8;
    return sweep.map((s, i) => {
      const px = pad + (s.review_rate / xmax) * (w - 2 * pad);
      const py = h - pad - (s.auto_error_rate / ymax) * (h - 2 * pad);
      return `${i ? 'L' : 'M'}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }

  errorReduction(u: any): number {
    return 1 - u.three_way.auto_errors / Math.max(u.binary.auto_errors, 1);
  }
}
