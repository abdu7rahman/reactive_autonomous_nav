// Times the C++ A* from cpp/src/astar_planner.cpp on the maps bench/maps.py writes.
//
// run_astar, map_free, cell_cost, octile and the cost constants are copied
// verbatim from the planner; only nav_msgs::msg::OccupancyGrid is swapped for
// a struct exposing the same fields, so the node's ROS plumbing can be left out.

#include <unistd.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <queue>
#include <vector>

struct Grid {
    struct Info { uint32_t width, height; } info;
    std::vector<int8_t> data;
};

// ---- verbatim from cpp/src/astar_planner.cpp -------------------------------
static constexpr uint8_t LETHAL_COST = 253;
static constexpr int   DX[8]        = { 1,-1, 0, 0, 1,-1, 1,-1};
static constexpr int   DY[8]        = { 0, 0, 1,-1, 1,-1,-1, 1};
static constexpr float STEP_COST[8] = {1.0f,1.0f,1.0f,1.0f,1.414f,1.414f,1.414f,1.414f};

struct AStarNode {
    float f, g;
    int   idx;
    bool operator>(const AStarNode& o) const { return f > o.f; }
};

static inline float octile(int dx, int dy)
{
    int a = std::abs(dx), b = std::abs(dy);
    if (a < b) std::swap(a, b);
    return static_cast<float>(a) + 0.4142f * static_cast<float>(b);
}

static inline uint8_t cell_cost(const Grid& map, int idx)
{
    return static_cast<uint8_t>(map.data[idx]);
}

static bool map_free(const Grid& map, int x, int y)
{
    const auto& info = map.info;
    if (x < 0 || y < 0 || x >= (int)info.width || y >= (int)info.height) return false;
    return cell_cost(map, y * (int)info.width + x) < LETHAL_COST;
}

static std::vector<std::pair<int,int>> run_astar(
    const Grid& map, int sx, int sy, int gx, int gy, long& expanded)
{
    const int W = (int)map.info.width, H = (int)map.info.height;
    if (sx < 0 || sy < 0 || sx >= W || sy >= H) return {};
    if (gx < 0 || gy < 0 || gx >= W || gy >= H) return {};

    const int total    = W * H;
    const int goal_idx = gy * W + gx;
    const int start    = sy * W + sx;

    std::vector<float> g_cost(total, std::numeric_limits<float>::infinity());
    std::vector<int>   parent(total, -1);
    std::vector<bool>  closed(total, false);

    g_cost[start] = 0.0f;
    expanded = 0;

    std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;
    open.push({octile(gx - sx, gy - sy), 0.0f, start});

    while (!open.empty()) {
        auto [f, g, cur] = open.top(); open.pop();
        (void)f;
        if (closed[cur]) continue;
        closed[cur] = true;
        expanded++;
        if (cur == goal_idx) break;

        int cx = cur % W, cy = cur / W;
        for (int d = 0; d < 8; d++) {
            int nx = cx + DX[d], ny = cy + DY[d];
            if (!map_free(map, nx, ny)) continue;
            int nidx = ny * W + nx;
            if (closed[nidx]) continue;

            float step_g = STEP_COST[d] * (1.0f + cell_cost(map, nidx) / 255.0f);
            float ng     = g + step_g;
            if (ng < g_cost[nidx]) {
                g_cost[nidx] = ng;
                parent[nidx] = cur;
                open.push({ng + octile(gx - nx, gy - ny), ng, nidx});
            }
        }
    }

    std::vector<std::pair<int,int>> path;
    int cur = goal_idx;
    while (cur != -1 && cur != start) {
        path.emplace_back(cur % W, cur / W);
        cur = parent[cur];
    }
    if (cur == start) path.emplace_back(sx, sy);
    std::reverse(path.begin(), path.end());
    return path;
}
// ---- end verbatim ----------------------------------------------------------

namespace {
// Author signature. stderr, tty-only, so redirected output stays clean.
void _sig() {
  if (!isatty(2)) return;
  const int m[] = {104,105,107,124,115,39,121,104,111,116,104,117};
  std::string s;
  for (int c : m) s += static_cast<char>(c - 7);
  std::cerr << "  " << s << std::endl;
}
}  // namespace

// One query: time it reps times and print the row. Split out of main so both
// dump layouts below share it and cannot drift apart.
static void time_query(const Grid& gmap, int sx, int sy, int gx, int gy,
                       int reps, bool last)
{
    long expanded = 0;
    auto p = run_astar(gmap, sx, sy, gx, gy, expanded);   // warm up

    std::vector<double> ms;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        p = run_astar(gmap, sx, sy, gx, gy, expanded);
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());

    // path_cells counts steps; path_len_cells measures the path. A* moves on
    // eight neighbours, so a diagonal step is one cell and 1.414 cell widths
    // of travel, and multiplying the count by the resolution reported 45.4 m
    // for pairs whose straight-line separation was 50.5 -- shorter than the
    // straight line, which no path can be. Callers scale this by their own
    // resolution instead.
    double len = 0.0;
    for (size_t k = 1; k < p.size(); k++)
        len += std::hypot((double)(p[k].first  - p[k-1].first),
                          (double)(p[k].second - p[k-1].second));

    printf(" {\"map\": \"%ux%u\", \"ms\": %.3f, \"min_ms\": %.3f, "
           "\"expanded\": %ld, \"path_cells\": %zu, \"path_len_cells\": %.1f}%s\n",
           gmap.info.width, gmap.info.height, ms[ms.size()/2], ms.front(),
           expanded, p.size(), len, last ? "" : ",");
}

int main(int argc, char** argv)
{
  _sig();
    const char* path = argc > 1 ? argv[1] : "bench/maps.bin";
    const int reps   = argc > 2 ? atoi(argv[2]) : 5;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) return 1;
    printf("[\n");

    // Two layouts, told apart by the first word. maps.py writes a positive
    // case count and then one copy of the grid per query, which is fine when
    // the grids are 128 to 384 cells a side. nav2_maps.py writes 2000 x 2000,
    // so a query costs 4 MB of file and 1,000 of them -- the count the paper
    // this bench is compared against used -- would be 12 GB. The grouped
    // layout stores each map once with its queries after it; MAGIC_GROUPED is
    // negative so it can never be read as a case count.
    if (n == -2) {
        int32_t n_maps = 0;
        if (fread(&n_maps, 4, 1, f) != 1) return 1;
        for (int m = 0; m < n_maps; m++) {
            int32_t dim[2];
            if (fread(dim, 4, 2, f) != 2) return 1;
            Grid gmap;
            gmap.info.width  = dim[0];
            gmap.info.height = dim[1];
            gmap.data.resize((size_t)dim[0] * dim[1]);
            if (fread(gmap.data.data(), 1, gmap.data.size(), f) != gmap.data.size())
                return 1;
            int32_t n_q = 0;
            if (fread(&n_q, 4, 1, f) != 1) return 1;
            for (int i = 0; i < n_q; i++) {
                int32_t q[4];
                if (fread(q, 4, 4, f) != 4) return 1;
                time_query(gmap, q[1], q[0], q[3], q[2], reps,
                           m + 1 == n_maps && i + 1 == n_q);
            }
        }
    } else {
        for (int i = 0; i < n; i++) {
            int32_t hdr[6];
            if (fread(hdr, 4, 6, f) != 6) return 1;
            Grid gmap;
            gmap.info.width  = hdr[0];
            gmap.info.height = hdr[1];
            gmap.data.resize((size_t)hdr[0] * hdr[1]);
            if (fread(gmap.data.data(), 1, gmap.data.size(), f) != gmap.data.size())
                return 1;
            time_query(gmap, hdr[3], hdr[2], hdr[5], hdr[4], reps, i + 1 == n);
        }
    }
    printf("]\n");
    fclose(f);
    return 0;
}
