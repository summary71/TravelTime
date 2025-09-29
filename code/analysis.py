import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
import json
import glob
from datetime import datetime, timezone
import pandas as pd
import math
from vrp_utils import manhattan_distance

plt.rcParams.update({
    'axes.titlesize': 24,      # default 12
    'axes.labelsize': 20,     # default 10
    'xtick.labelsize': 16,    # default 8
    'ytick.labelsize': 16,    # default 8
    'legend.fontsize': 16,    # default 8
    'figure.titlesize': 24    # default 12
})

# Load JSON files
#json_files = glob.glob("../VRPdataSmall/*.json")
json_files = glob.glob("../VRPdata/*.json")

# Initialize all containers
total_routes = 0
total_segments = 0
unique_segments = set()
destinations = []
unique_destinations = set()
segment_counts_per_route = []
collection_dates = set()

date_counts = Counter()
dest_counts = Counter()
segment_counts_counter = Counter()
route_length_distribution = Counter()
geodesic_distances_km = []
manhattan_distances_km = []
# Define geodesic function
def geodesic_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

geodesic_vs_time = []
manhattan_vs_time = []
# Process each file ONCE
for file in json_files:
    try:
        with open(file, 'r') as f:
            segments = json.load(f)
            total_routes += 1
            segment_counts_per_route.append(len(segments))
            total_segments += len(segments)
            route_length_distribution[len(segments)] += 1

            if segments:
                first_seg = segments[0]
                ts = first_seg.get("departureTime")
                if ts:
                    date = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                    collection_dates.add(date)
                    date_counts[date] += 1

            for seg in segments:
                from_coord = (round(seg["from"]["lat"], 6), round(seg["from"]["lng"], 6))
                to_coord = (round(seg["to"]["lat"], 6), round(seg["to"]["lng"], 6))

                if from_coord == to_coord:
                    continue
                if seg["from"]["lat"] < 33.6 or seg["to"]["lat"] < 33.6:
                    continue
                if seg["from"]["lng"] > 130 or seg["to"]["lng"] > 130:
                    continue

                unique_segments.add((from_coord, to_coord))
                destinations.append(to_coord)
                unique_destinations.add(to_coord)
                dest_counts[to_coord] += 1

                segment = f"{from_coord} → {to_coord}"
                segment_counts_counter[segment] += 1

                lat1, lon1 = seg["from"]["lat"], seg["from"]["lng"]
                lat2, lon2 = seg["to"]["lat"], seg["to"]["lng"]
                dist_km = geodesic_distance(lat1, lon1, lat2, lon2)
                geodesic_distances_km.append(dist_km)
                dep = seg.get("departureTime")
                arr = seg.get("arrivalTime")
                if dep is not None and arr is not None:
                    travel_time = (arr - dep) / 1000  # milliseconds to seconds
                    if travel_time > 0:
                        geodesic_vs_time.append((dist_km, travel_time))
                        manhattan_dist = manhattan_distance(lat1, lon1, lat2, lon2)
                        manhattan_distances_km.append(manhattan_dist)
                        manhattan_vs_time.append((manhattan_dist, travel_time))

    except Exception as e:
        continue

# Summary statistics
num_unique_segments = len(unique_segments)
num_destinations = len(destinations)
num_unique_destinations = len(unique_destinations)
avg_segments_per_route = sum(segment_counts_per_route) / total_routes if total_routes > 0 else 0
sorted_dates = sorted(collection_dates)

print(f"Max Geodesic Distance: {max(geodesic_distances_km)} km ")
print(f"Max Manhattan Distance: {max(manhattan_distances_km)} km")

df_result = pd.DataFrame({
    "Metric": [
        "Number of Routes",
        "Number of Segments",
        "Number of Unique Segments",
        "Number of Destinations",
        "Number of Unique Destinations",
        "Average Number of Segments per Route",
        "Collection Dates"
    ],
    "Value": [
        total_routes,
        total_segments,
        num_unique_segments,
        num_destinations,
        num_unique_destinations,
        round(avg_segments_per_route, 2),
        f"{sorted_dates[0]} ~ {sorted_dates[-1]}" if sorted_dates else "N/A"
    ]
})

print(df_result)
df_result.to_csv("driving_dataset_summary.csv", index=False)

# === Graphs ===
# Graph 1: Route Counts per Day
pdf_route = "./graphs/routecount.pdf"
with PdfPages(pdf_route) as pdf:
    plt.figure(figsize=(10, 6))
    plt.bar(date_counts.keys(), date_counts.values(), color='skyblue')
    plt.xticks([], [])
    plt.title("Route Counts per Day")
    plt.xlabel("Date")
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Graph 2: Top 200 Segments
pdf_route = "./graphs/topsegs.pdf"
with PdfPages(pdf_route) as pdf:
    top_segments = segment_counts_counter.most_common(100)
    labels, values = zip(*top_segments)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='green')
    plt.xticks([], [])
    plt.title("Top 100 Most Appeared Segments")
    plt.xlabel("Segment (from → to)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Graph 3: Route Lengths
pdf_route = "./graphs/routelengths.pdf"
with PdfPages(pdf_route) as pdf:
    lengths, counts = zip(*sorted(route_length_distribution.items()))
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='purple')
    plt.title("Distribution of Route Lengths")
    plt.xlabel("Route Length (number of segments)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Graph 4: Top 100 Destinations
pdf_route = "./graphs/topdest.pdf"
with PdfPages(pdf_route) as pdf:
    top_dest = dest_counts.most_common(100)
    labels, values = zip(*[(f"{lat:.3f},{lng:.3f}", count) for (lat, lng), count in top_dest])
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='orange')
    plt.xticks([], [])
    plt.title("Top 100 Most Appeared Destinations")
    plt.xlabel("Destination")
    plt.ylabel("Frequency")
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Graph 5: geodesic Histogram
bins = range(0, int(max(geodesic_distances_km)) + 5, 5)
pdf_route = "./graphs/geodesic_histogram.pdf"
with PdfPages(pdf_route) as pdf:
    plt.figure(figsize=(10, 6))
    plt.hist(geodesic_distances_km, bins=bins, density=True, edgecolor='black', color='skyblue')
    plt.title("Relative Frequency of Geodesic Distances (5 km bins)")
    plt.xlabel("geodesic Distance (km)")
    plt.ylabel("Relative Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# Graph 6: geodesic Histogram (Normalized Relative Frequency)

# Step 1: Compute histogram manually
counts, bin_edges = np.histogram(geodesic_distances_km, bins=bins)
total_counts = sum(counts)
relative_freq = counts / total_counts

# Step 2: Plot manually normalized histogram
pdf_route = "./graphs/geodesic_histogram_normalized.pdf"
with PdfPages(pdf_route) as pdf:
    plt.figure(figsize=(13, 6))
    plt.bar(bin_edges[:-1], relative_freq, width=5, edgecolor='black', color='skyblue', align='edge')
    plt.title("Normalized Relative Frequency of Geodesic Distances (5 km bins)")
    plt.xlabel("Geodesic Distance (km)")
    plt.ylabel("Relative Frequency (Sum=1)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("✅ 모든 분석 + Normalized geodesic Histogram 완료 → graphs/ 디렉토리에 PDF 파일 생성됨.")

# === Geodesic Distance vs Travel Time Scatter Plot ===
if geodesic_vs_time:
    dists, times = zip(*geodesic_vs_time)
    dists = np.array(dists)
    times = np.array(times)
    np.random.seed(42)  # For reproducibility
    if len(dists) > 10:
        idx = np.random.choice(len(dists), size=len(dists)//20, replace=False)
        dists = dists[idx]
        times = times[idx]
    pdf_route = "./graphs/geodesic_vs_time_scatter.pdf"
    with PdfPages(pdf_route) as pdf:
        plt.figure(figsize=(10, 6))
        plt.scatter(dists, times, alpha=0.3, s=8, color='blue')
        plt.title("Geodesic Distance vs Travel Time per Segment")
        plt.xlabel("Geodesic Distance (km)")
        plt.ylabel("Travel Time (seconds)")
        plt.xlim([0, 5])
        plt.ylim([0, 1800])
        # Fit and plot linear regression line
        try:
            slope, intercept = np.polyfit(dists, times, 1)
            x_min, x_max = plt.xlim()
            x_fit = np.linspace(x_min, x_max, 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, color='black', linewidth=2, label=f"Linear fit: y={slope:.1f}x+{intercept:.0f}")
            plt.legend()
        except Exception:
            pass
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print(f"✅ Geodesic Distance vs Travel Time scatter plot saved to {pdf_route}")

# === Manhattan Distance vs Travel Time Scatter Plot ===
if manhattan_vs_time:
    manhattan_dists, times = zip(*manhattan_vs_time)
    manhattan_dists = np.array(manhattan_dists)
    times = np.array(times)
    np.random.seed(42)  # For reproducibility
    if len(manhattan_dists) > 10:
        idx = np.random.choice(len(manhattan_dists), size=len(manhattan_dists)//20, replace=False)
        manhattan_dists = manhattan_dists[idx]
        times = times[idx]
    pdf_route = "./graphs/manhattan_vs_time_scatter.pdf"
    with PdfPages(pdf_route) as pdf:
        plt.figure(figsize=(10, 6))
        plt.scatter(manhattan_dists, times, alpha=0.3, s=8, color='red')
        plt.title("Manhattan Distance vs Travel Time per Segment")
        plt.xlabel("Manhattan Distance (km)")
        plt.ylabel("Travel Time (seconds)")
        plt.xlim([0, 5])
        plt.ylim([0, 1800])
        # Fit and plot linear regression line
        try:
            slope, intercept = np.polyfit(manhattan_dists, times, 1)
            x_min, x_max = plt.xlim()
            x_fit = np.linspace(x_min, x_max, 100)
            y_fit = slope * x_fit + intercept
            plt.plot(x_fit, y_fit, color='black', linewidth=2, label=f"Linear fit: y={slope:.1f}x+{intercept:.0f}")
            plt.legend()
        except Exception:
            pass
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    print(f"✅ Manhattan Distance vs Travel Time scatter plot saved to {pdf_route}")

# === Merged Cumulative Distribution of Segment and Destination Appearance Counts ===
# Compute CDF for segments
all_seg_counts = np.array(list(segment_counts_counter.values()))
seg_sorted = np.sort(all_seg_counts)
seg_cdf = np.arange(1, len(seg_sorted)+1) / len(seg_sorted)
seg_mask = seg_sorted < 101
seg_sorted = seg_sorted[seg_mask]
seg_cdf = seg_cdf[:np.sum(seg_mask)]

# Compute CDF for destinations
all_dest_counts = np.array(list(dest_counts.values()))
dest_sorted = np.sort(all_dest_counts)
dest_cdf = np.arange(1, len(dest_sorted)+1) / len(dest_sorted)
dest_mask = dest_sorted < 101
dest_sorted = dest_sorted[dest_mask]
dest_cdf = dest_cdf[:np.sum(dest_mask)]

pdf_route = "./graphs/merged_cumulative.pdf"
with PdfPages(pdf_route) as pdf:
    plt.figure(figsize=(10, 6))
    plt.step(seg_sorted, seg_cdf, where='post', color='blue', label='Segments')
    plt.step(dest_sorted, dest_cdf, where='post', color='orange', label='Destinations')
    plt.title("Cumulative Distribution of Appearance Counts")
    plt.xlabel("Appearance Frequency")
    plt.ylabel("Cumulative Probability")
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.legend()
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
