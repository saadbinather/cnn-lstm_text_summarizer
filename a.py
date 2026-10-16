import matplotlib.pyplot as plt

# Tumhare exact results
configs = ['Layer 1 (1-gram)', 'Stacked (1-2)', 'Stacked (1-3)', 'Multi-Kernel (1-4)']
rouge1 = [0.6262, 0.6262, 0.6780, 0.7143]

plt.figure(figsize=(8, 4.5))
plt.plot(configs, rouge1, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='ROUGE-1')

plt.title('Ablation Study: Stacked CNN Kernel Progression', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Configurations', fontsize=10)
plt.ylabel('ROUGE-1 Score', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(0.60, 0.73)

# Highlight proposed model
plt.annotate('Proposed (0.7143)', xy=(3, 0.7143), xytext=(2.1, 0.65),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
             fontsize=9, fontweight='semibold', backgroundcolor='white')

plt.tight_layout()
plt.savefig('ablation_results_graph.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved as 'ablation_results_graph.png'!")
plt.show()