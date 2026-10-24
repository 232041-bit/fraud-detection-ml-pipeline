import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResponse {
  prediction: string | number;
  task: string;
  model?: string;
  fraud_probability?: number;
  threshold?: number;
  probabilities?: number[];
  disclaimer?: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);
  predict(payload: Record<string, string | number>, model: string): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(`/api/predict?model=${model}`, payload);
  }
}
