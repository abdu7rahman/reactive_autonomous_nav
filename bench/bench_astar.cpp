// Times the C++ A* from cpp/src/astar_planner.cpp on the maps bench/maps.py writes.
//
// run_astar, map_free, cell_cost, octile and the cost constants are copied
// verbatim from the planner; only nav_msgs::msg::OccupancyGrid is swapped for
// a struct exposing the same fields, so the node's ROS plumbing can be left out.

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

int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : "bench/maps.bin";
    const int reps   = argc > 2 ? atoi(argv[2]) : 5;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }

    int32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) return 1;
    printf("[\n");
    for (int i = 0; i < n; i++) {
        int32_t hdr[6];
        if (fread(hdr, 4, 6, f) != 6) return 1;
        Grid gmap;
        gmap.info.width  = hdr[0];
        gmap.info.height = hdr[1];
        gmap.data.resize((size_t)hdr[0] * hdr[1]);
        if (fread(gmap.data.data(), 1, gmap.data.size(), f) != gmap.data.size()) return 1;

        const int sy = hdr[2], sx = hdr[3], gy = hdr[4], gx = hdr[5];
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
        printf(" {\"map\": \"%dx%d\", \"ms\": %.3f, \"min_ms\": %.3f, "
               "\"expanded\": %ld, \"path_cells\": %zu}%s\n",
               hdr[0], hdr[1], ms[ms.size()/2], ms.front(), expanded, p.size(),
               i + 1 < n ? "," : "");
    }
    printf("]\n");
    fclose(f);
    return 0;
}
