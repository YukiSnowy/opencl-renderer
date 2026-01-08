float edgeFunction(float2 a, float2 b, float2 c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

float3 unpackColor(uint packed_color) {
    return (float3)(((packed_color >> 16) & 0xFF) / 255.0f,
                    ((packed_color >> 8) & 0xFF) / 255.0f,
                    (packed_color & 0xFF) / 255.0f);
}

uint packColor(float3 unpacked_color) {
    uint r = (uint)(unpacked_color.x * 255.0f);
    uint g = (uint)(unpacked_color.y * 255.0f);
    uint b = (uint)(unpacked_color.z * 255.0f);
    return (r << 16) | (g << 8) | b;
}


__kernel void rasterize_interpolated_triangle(
    __global uint* output,
    int width, int height,
    float2 v0_pos, float2 v1_pos, float2 v2_pos,
    uint v0_color, uint v1_color, uint v2_color
) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    output[y * width + x] = 0x000000;

    float2 p = (float2)(x + 0.5f, y + 0.5f);

    float area = edgeFunction(v0_pos, v1_pos, v2_pos);

    if (area == 0.0f) return;

    float w0 = edgeFunction(v1_pos, v2_pos, p);
    float w1 = edgeFunction(v2_pos, v0_pos, p);
    float w2 = edgeFunction(v0_pos, v1_pos, p);

    if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        w0 /= area;
        w1 /= area;
        w2 /= area;

        float3 c0 = unpackColor(v0_color);
        float3 c1 = unpackColor(v1_color);
        float3 c2 = unpackColor(v2_color);

        float3 interpolated_color = c0 * w0 + c1 * w1 + c2 * w2;

        output[y * width + x] = packColor(interpolated_color);
    }
}