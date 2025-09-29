import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import sys

# Argument parser 설정
parser = argparse.ArgumentParser(description="Plot CDF of reldiffs from multiple CSV files and save as PDF.")
parser.add_argument('file_alias', nargs='+', help='Pairs of CSV filename and alias, e.g., file1.csv ModelA file2.csv ModelB ...')
parser.add_argument('--output', type=str, default='reldiffs_cdf_plot.pdf', help='Output PDF filename (default: reldiffs_cdf_plot.pdf)')
parser.add_argument('--xmax', type=float, default=1.0, help='Maximum value for x-axis range (default: 1.0)')

args = parser.parse_args()

# 폰트 크기
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# 입력 검증: 짝수 개여야 함 (파일명, 별명)
if len(args.file_alias) % 2 != 0:
    print("Error: You must provide pairs of CSV filename and alias (even number of arguments).", args.file_alias)
    sys.exit(1)

# file_alias_list 구성
file_alias_list = []
for i in range(0, len(args.file_alias), 2):
    filename = args.file_alias[i]
    alias = args.file_alias[i+1]
    file_alias_list.append((filename, alias))

# Plot
plt.figure(figsize=(10, 6))

for filename, alias in file_alias_list:
    # CSV 파일 읽기
    df = pd.read_csv(filename)
    
    # reldiffs 컬럼 추출
    reldiffs = df['reldiffs'].dropna().values
    
    # CDF 계산
    sorted_reldiffs = np.sort(reldiffs)
    cdf = np.arange(1, len(sorted_reldiffs)+1) / len(sorted_reldiffs)
    
    # 그래프 그리기
    plt.plot(sorted_reldiffs, cdf, label=alias)

# 그래프 설정
plt.xlim(0, args.xmax)
plt.ylim(0, 1)
plt.xlabel('Relative Error', fontsize=18)
plt.ylabel('Cumulative Probability', fontsize=18)
plt.title('Cumulative Distribution of Relative Error', fontsize=20, fontweight='bold')
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()

# PDF 저장
plt.savefig(args.output)
print(f"PDF saved to: {args.output}")
