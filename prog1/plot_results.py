#!/usr/bin/env python3
"""
Script per generare grafici dai risultati del benchmark di moltiplicazione matrici.
Viene chiamato automaticamente da ./matmul benchmark
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main():
    # Verifica che esista il file dei risultati
    if not os.path.exists('output/results.csv'):
        print("❌ Errore: output/results.csv non trovato. Esegui prima il benchmark.")
        sys.exit(1)
    
    # Crea cartella per i grafici se non esiste
    os.makedirs('figures', exist_ok=True)
    
    # Carica i dati
    df = pd.read_csv('output/results.csv')
    
    # Filtra righe valide
    df = df[df['Speedup'] > 0]
    df = df[df['Speedup'] < 100]
    
    print(f"✅ Dati caricati: {len(df)} righe")
    
    # =========================================================================
    # FIGURA 1: Speedup al variare del numero di thread
    # =========================================================================
    fig1, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for idx, n in enumerate([512, 1024, 2048]):
        ax = axes[idx]
        df_n = df[df['N'] == n]
        
        for schedule in ['static', 'dynamic', 'guided']:
            df_sched = df_n[df_n['Schedule'] == schedule]
            best_by_thread = []
            for t in sorted(df_sched['Threads'].unique()):
                df_t = df_sched[df_sched['Threads'] == t]
                best_speedup = df_t['Speedup'].max()
                best_by_thread.append((t, best_speedup))
            
            if best_by_thread:
                threads, speedups = zip(*best_by_thread)
                ax.plot(threads, speedups, 'o-', linewidth=2, markersize=8, 
                       label=schedule.capitalize())
        
        max_thread = df_n['Threads'].max()
        ax.plot([1, max_thread], [1, max_thread], 'k--', alpha=0.5, label='Ideale')
        ax.set_xlabel('Numero di thread')
        ax.set_title(f'N = {n}')
        ax.set_xlim(0.5, max_thread + 0.5)
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel('Speedup')
    fig1.suptitle('Speedup al variare del numero di thread', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    fig1.savefig('figures/fig1_speedup_threads.pdf', bbox_inches='tight')
    fig1.savefig('figures/fig1_speedup_threads.png', bbox_inches='tight', dpi=150)
    print("  ✅ fig1_speedup_threads.pdf/png")
    plt.close(fig1)
    
    # =========================================================================
    # FIGURA 2: Efficienza parallela
    # =========================================================================
    fig2, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for idx, n in enumerate([512, 1024, 2048]):
        ax = axes[idx]
        df_n = df[df['N'] == n]
        
        for schedule in ['static', 'dynamic', 'guided']:
            df_sched = df_n[df_n['Schedule'] == schedule]
            best_by_thread = []
            for t in sorted(df_sched['Threads'].unique()):
                df_t = df_sched[df_sched['Threads'] == t]
                best_speedup = df_t['Speedup'].max()
                efficiency = best_speedup / t if t > 0 else 0
                best_by_thread.append((t, efficiency))
            
            if best_by_thread:
                threads, effs = zip(*best_by_thread)
                ax.plot(threads, effs, 'o-', linewidth=2, markersize=8, 
                       label=schedule.capitalize())
        
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideale')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Numero di thread')
        ax.set_title(f'N = {n}')
        ax.set_xlim(0.5, 8.5)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel('Efficienza (Speedup / Threads)')
    fig2.suptitle('Efficienza parallela al variare del numero di thread', fontsize=16, fontweight='bold')
    fig2.tight_layout()
    fig2.savefig('figures/fig2_efficiency.pdf', bbox_inches='tight')
    fig2.savefig('figures/fig2_efficiency.png', bbox_inches='tight', dpi=150)
    print("  ✅ fig2_efficiency.pdf/png")
    plt.close(fig2)
    
    # =========================================================================
    # FIGURA 3: Shared vs Private
    # =========================================================================
    fig3, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, n in enumerate([512, 1024, 2048]):
        ax = axes[idx]
        df_n = df[(df['N'] == n) & (df['Schedule'] == 'dynamic')]
        
        for variant in ['shared', 'private']:
            df_var = df_n[df_n['Variant'] == variant]
            best_by_thread = []
            for t in sorted(df_var['Threads'].unique()):
                df_t = df_var[df_var['Threads'] == t]
                best_speedup = df_t['Speedup'].max()
                best_by_thread.append((t, best_speedup))
            
            if best_by_thread:
                threads, speedups = zip(*best_by_thread)
                marker = 's' if variant == 'shared' else 'o'
                linestyle = '-' if variant == 'shared' else '--'
                ax.plot(threads, speedups, marker=marker, linestyle=linestyle, 
                       linewidth=2, markersize=8, label=variant.capitalize())
        
        ax.set_xlabel('Numero di thread')
        ax.set_title(f'N = {n}')
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel('Speedup')
    fig3.suptitle('Confronto Shared vs Private (scheduler dynamic)', fontsize=16, fontweight='bold')
    fig3.tight_layout()
    fig3.savefig('figures/fig3_shared_vs_private.pdf', bbox_inches='tight')
    fig3.savefig('figures/fig3_shared_vs_private.png', bbox_inches='tight', dpi=150)
    print("  ✅ fig3_shared_vs_private.pdf/png")
    plt.close(fig3)
    
    # =========================================================================
    # FIGURA 4: Effetto chunk size
    # =========================================================================
    fig4, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    chunks = [1, 8, 32, 128]
    
    for idx, n in enumerate([512, 1024, 2048]):
        ax = axes[idx]
        df_n = df[(df['N'] == n) & (df['Schedule'] == 'static') & (df['Threads'] == 8)]
        
        x = np.arange(len(chunks))
        width = 0.35
        
        shared_speedups = []
        private_speedups = []
        
        for chunk in chunks:
            df_chunk = df_n[df_n['ChunkSize'] == chunk]
            
            shared_val = df_chunk[df_chunk['Variant'] == 'shared']['Speedup'].values
            shared_speedups.append(shared_val[0] if len(shared_val) > 0 else 0)
            
            private_val = df_chunk[df_chunk['Variant'] == 'private']['Speedup'].values
            private_speedups.append(private_val[0] if len(private_val) > 0 else 0)
        
        ax.bar(x - width/2, shared_speedups, width, label='Shared', color='steelblue')
        ax.bar(x + width/2, private_speedups, width, label='Private', color='coral')
        ax.set_xlabel('Chunk size')
        ax.set_title(f'N = {n}')
        ax.set_xticks(x)
        ax.set_xticklabels(chunks)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    
    axes[0].set_ylabel('Speedup')
    fig4.suptitle('Effetto del chunk size (static scheduler, 8 thread)', fontsize=16, fontweight='bold')
    fig4.tight_layout()
    fig4.savefig('figures/fig4_chunk_effect.pdf', bbox_inches='tight')
    fig4.savefig('figures/fig4_chunk_effect.png', bbox_inches='tight', dpi=150)
    print("  ✅ fig4_chunk_effect.pdf/png")
    plt.close(fig4)
    
    # =========================================================================
    # FIGURA 5: Confronto scheduler
    # =========================================================================
    fig5, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, n in enumerate([512, 1024, 2048]):
        ax = axes[idx]
        df_n = df[(df['N'] == n) & (df['Variant'] == 'private')]
        
        for schedule in ['static', 'dynamic', 'guided']:
            df_sched = df_n[df_n['Schedule'] == schedule]
            best_by_thread = []
            for t in sorted(df_sched['Threads'].unique()):
                df_t = df_sched[df_sched['Threads'] == t]
                best_speedup = df_t['Speedup'].max()
                best_by_thread.append((t, best_speedup))
            
            if best_by_thread:
                threads, speedups = zip(*best_by_thread)
                marker_map = {'static': 's', 'dynamic': 'o', 'guided': '^'}
                ax.plot(threads, speedups, marker=marker_map[schedule], linestyle='-',
                       linewidth=2, markersize=8, label=schedule.capitalize())
        
        ax.set_xlabel('Numero di thread')
        ax.set_title(f'N = {n}')
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[0].set_ylabel('Speedup')
    fig5.suptitle('Confronto scheduler (variante private)', fontsize=16, fontweight='bold')
    fig5.tight_layout()
    fig5.savefig('figures/fig5_scheduler_comparison.pdf', bbox_inches='tight')
    fig5.savefig('figures/fig5_scheduler_comparison.png', bbox_inches='tight', dpi=150)
    print("  ✅ fig5_scheduler_comparison.pdf/png")
    plt.close(fig5)
    
    # =========================================================================
    # TABELLE LaTeX (stampate a schermo)
    # =========================================================================
    print("\n" + "="*60)
    print("📊 TABELLE LaTeX pronte per il report:")
    print("="*60)
    
    # Tabella riassuntiva
    print("\n\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Riepilogo dei migliori speedup per dimensione.}")
    print("\\label{tab:summary}")
    print("\\begin{tabular}{rrrrrr}")
    print("\\toprule")
    print("N & Threads & Schedule & Chunk & Variante & Speedup \\\\")
    print("\\midrule")
    
    for n in [512, 1024, 2048]:
        df_n = df[df['N'] == n]
        best_row = df_n.loc[df_n['Speedup'].idxmax()]
        print(f"{int(best_row['N'])} & {int(best_row['Threads'])} & {best_row['Schedule']} & "
              f"{int(best_row['ChunkSize'])} & {best_row['Variant']} & {best_row['Speedup']:.2f}$\\times$ \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    print("\n✅ Tutti i grafici salvati in 'figures/'")
    return 0

if __name__ == "__main__":
    sys.exit(main())