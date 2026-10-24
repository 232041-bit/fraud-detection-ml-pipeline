import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { FEATURES, Feature, SAMPLE_FRAUD, SAMPLE_LEGIT } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields: Feature[] = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  readonly models = [
    { id: 'random_forest', label: 'Random Forest' },
    { id: 'logistic_regression', label: 'Logistic Regression' },
  ];
  selectedModel = 'random_forest';

  values: Record<string, string | number | null> = {};
  readonly loading = signal(false);
  readonly error = signal('');
  readonly result = signal<PredictionResponse | null>(null);

  private api = inject(PredictionService);

  constructor() {
    for (const field of this.fields) this.values[field.name] = null;
  }

  loadSample(kind: 'legit' | 'fraud') {
    const sample = kind === 'legit' ? SAMPLE_LEGIT : SAMPLE_FRAUD;
    for (const field of this.fields) this.values[field.name] = sample[field.name];
    this.error.set('');
    this.result.set(null);
  }

  clearForm() {
    for (const field of this.fields) this.values[field.name] = null;
    this.error.set('');
    this.result.set(null);
  }

  percent(p?: number): string {
    return p === undefined ? '' : (p * 100).toFixed(2) + '%';
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
    this.api.predict(payload, this.selectedModel).subscribe({
      next: (res) => { this.result.set(res); this.loading.set(false); },
      error: (err) => {
        this.error.set(`API error ${err.status || ''}: ` +
          (err.error?.error || err.message || 'Check Flask terminal'));
        this.loading.set(false);
      },
    });
  }
}
