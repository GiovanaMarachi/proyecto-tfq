#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plot_styles
from datetime import datetime

def generate_all_plots(csv_path, history_df=None, output_base_dir='outputs/reports'):
    # 1. Crear carpeta dinámica con fecha y hora
    timestamp_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_dir = os.path.join(output_base_dir, timestamp_folder)
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Cargar datos
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    real = df['Real'].values
    pred = df['Predicho'].values
    dates = df['Timestamp'].values
    
    plot_styles.set_pub_style()
    colors = plot_styles.get_pub_colors()
    date_fmt = mdates.DateFormatter('%d-%m-%y') # Solo fecha como pidió el usuario
    
    print(f"🎨 Generando gráficas en: {target_dir}")

    # --- A. Gráfico de LOSS (si hay historial) ---
    if history_df is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history_df['loss'], label='Entrenamiento', color=colors['real'])
        if 'val_loss' in history_df.columns:
            ax.plot(history_df['val_loss'], label='Validación', color=colors['pred'], linestyle='--')
        ax.set_title('Historial de Pérdida (Loss History)')
        ax.set_xlabel('Época')
        ax.set_ylabel('MSE')
        ax.set_yscale('log') # Escala logarítmica para ver mejor la bajada
        ax.legend()
        plot_styles.save_publication_figures('loss_history', target_dir)
        plt.close(fig)
        print("   ✅ Loss plot guardado.")

    # --- B. Gráfico PREDICTION (Completo) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, real, label='Precio Real', color=colors['real'], linewidth=1)
    ax.plot(dates, pred, label='Predicción', color=colors['pred'], linewidth=0.8, alpha=0.9)
    ax.xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()
    ax.set_title('Bitcoin Prediction - Full Test Set')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    plot_styles.save_publication_figures('prediction_full', target_dir)
    plt.close(fig)
    print("   ✅ Prediction Full guardado.")

    # --- C. Gráfico PREDICTION_DETAILED (Últimos 150 puntos) ---
    n_det = 150
    if len(real) > n_det:
        fig, ax = plt.subplots(figsize=(12, 6))
        z_dates = dates[-n_det:]
        z_real = real[-n_det:]
        z_pred = pred[-n_det:]
        
        ax.plot(z_dates, z_real, label='Real', color=colors['real'], marker='o', markersize=4, linewidth=1)
        ax.plot(z_dates, z_pred, label='Predicción', color=colors['pred'], marker='x', markersize=5, linestyle='--', linewidth=1)
        
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        ax.set_title(f'Prediction Detailed (Last {n_det} points)')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        plot_styles.save_publication_figures('prediction_detailed', target_dir)
        plt.close(fig)
        print(f"   ✅ Detailed Zoom ({n_det} pts) guardado.")

    # --- D. Gráfico PREDICTION_BEST_ZOOM (Mejores 100 puntos) ---
    win = 100
    if len(real) > win:
        mae = np.abs(real - pred)
        best_idx = 0
        min_mae = float('inf')
        for i in range(len(mae) - win):
            m = np.mean(mae[i:i+win])
            if m < min_mae:
                min_mae = m
                best_idx = i
        
        fig, ax = plt.subplots(figsize=(12, 6))
        b_dates = dates[best_idx : best_idx+win]
        b_real = real[best_idx : best_idx+win]
        b_pred = pred[best_idx : best_idx+win]
        
        ax.plot(b_dates, b_real, label='Real', color=colors['real'], marker='o', markersize=4)
        ax.plot(b_dates, b_pred, label='Predicción', color=colors['pred'], marker='x', markersize=5, linestyle='--')
        
        ax.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()
        ax.set_title(f'Best Match Zoom (Window {win} pts, MAE: {min_mae:.2f})')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        plot_styles.save_publication_figures('prediction_best_zoom', target_dir)
        plt.close(fig)
        print(f"   ✅ Best Zoom ({win} pts) guardado.")

    return target_dir

def main():
    # Para compatibilidad con ejecuciones directas
    csv_path = 'outputs/prediction_results.csv'
    generate_all_plots(csv_path)

if __name__ == '__main__':
    main()
