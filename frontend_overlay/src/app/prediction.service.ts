import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export type Decision = 'APPROVE' | 'MANUAL_REVIEW' | 'FLAG_FRAUD';

export interface PredictionResponse {
  prediction: string | number;
  task: string;
  model?: string;
  fraud_probability?: number;
  threshold?: number;
  probabilities?: number[];
  disclaimer?: string;
  // Novelty 2 - uncertainty-aware three-way decision
  decision?: Decision;
  decision_reasons?: string[];
  uncertainty?: {
    method: string;
    members: number;
    std: number;
    predictive_entropy: number;
    aleatoric_entropy: number;
    mutual_information: number;
    epistemic_limit: number | null;
    grey_zone: [number, number];
  };
  // Novelty 1 - drift-aware threshold used for this decision
  drift?: {
    adaptive_threshold: number;
    calibrated_threshold: number;
    original_threshold: number;
    fair_mode: boolean;
    age_group: string;
    recalibrations: number;
  };
}

export interface DriftFeature {
  feature: string;
  psi: number;
  ks_stat: number | null;
  ks_p: number | null;
  severity: string;
}

export interface Perf { fpr: number | null; recall: number | null; }

export interface DriftLogEntry {
  batch: number;
  n: number;
  month: number | null;
  drift_detected: boolean;
  severity: string;
  max_psi: number;
  score_psi: number | null;
  threshold_before: number;
  threshold_after: number;
  group_thresholds: Record<string, number>;
  recalibrated: boolean;
  reason: string | null;
  performance_before: Perf | null;
  performance_after: Perf | null;
  decisions: Record<Decision, number>;
}

export interface PolicyState {
  mode: string;
  fair: boolean;
  target_fpr: number;
  threshold: number;
  base_threshold: number;
  group_thresholds: Record<string, number>;
  recalibrations: number;
}

export interface DriftBatchResponse {
  model: string;
  report: {
    n: number; drift_detected: boolean; severity: string; feature_drift: boolean;
    score_drift: boolean; max_psi: number; drifted_features: string[];
    score: DriftFeature | null; top_features: DriftFeature[];
  };
  update: DriftLogEntry;
  policy: PolicyState;
}

export interface DemoBatch {
  month: number;
  records: Record<string, string | number>[];
  labels: number[];
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  predict(payload: Record<string, string | number>, model: string, fair = false):
      Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(
      `/api/predict?model=${model}&fair=${fair ? 1 : 0}`, payload);
  }

  demoBatch(month: number, n = 1500): Observable<DemoBatch> {
    return this.http.get<DemoBatch>(`/api/drift/demo-batch?month=${month}&n=${n}`);
  }

  sendBatch(model: string, batch: DemoBatch, withLabels: boolean, fair: boolean):
      Observable<DriftBatchResponse> {
    return this.http.post<DriftBatchResponse>(
      `/api/drift/batch?model=${model}&fair=${fair ? 1 : 0}`,
      { records: batch.records, labels: withLabels ? batch.labels : null, month: batch.month });
  }

  driftStatus(model: string): Observable<{ policy: PolicyState; log: DriftLogEntry[] }> {
    return this.http.get<{ policy: PolicyState; log: DriftLogEntry[] }>(
      `/api/drift/status?model=${model}`);
  }

  resetDrift(model: string, fair: boolean): Observable<{ policy: PolicyState }> {
    return this.http.post<{ policy: PolicyState }>(
      `/api/drift/reset?model=${model}&fair=${fair ? 1 : 0}`, {});
  }

  /** Static experiment summary (bundled asset -> also works on GitHub Pages). */
  summary(): Observable<any> {
    return this.http.get<any>('novelty_summary.json');
  }
}
