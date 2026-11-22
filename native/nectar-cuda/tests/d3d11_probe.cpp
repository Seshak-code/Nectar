// d3d11-probe: an ordinary Direct3D 11 program, to run through DXVK and Nectar's Vulkan driver.
// Each frame it does what a game's does: updates a constant buffer with Map(DISCARD) before each of
// many draws, samples a texture through compiled shaders, optionally re-uploads a texture, and
// presents. It says which adapter it got, how fast frames went, and checks the colours on screen.
//
// Build, in a Visual Studio x64 prompt:
//   cl /nologo /EHsc /O2 d3d11_probe.cpp /link d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib
// Run through DXVK and Nectar: DXVK's x64 d3d11.dll and dxgi.dll beside it, VK_DRIVER_FILES (or
// Nectar installed), and DXVK_FILTER_DEVICE_NAME=(Nectar).
//
// Usage: d3d11_probe [--frames N] [--draws N] [--size WxH] [--upload-kib N] [--upload-kind K]
//                    [--new-textures 0|1] [--burst-frame N] [--upload-file PATH --upload-side N]
//                    [--srgb 0|1] [--show 0|1] [--shots PREFIX,MS,MS...]
//
// --upload-kind says what the upload holds, as a game's memory writes differ:
//   same    the same each frame but for one texel
//   floats  floats that all change a little each frame, as skinned vertices or matrices do
//   noise   bytes that won't compress, as streamed compressed textures are
//   bc7     the same, as a BC7 texture and all its mips
// --new-textures 1 uploads to a new texture each time, as loading does, not the same one, as animating
// does. What the GPU ends up holding is read back and compared, byte for byte. --burst-frame N uploads
// only on that frame, to see how long the game stops for it. --upload-file uploads a BC7 mip chain from a
// file, for pictures rather than noise. --show 1 draws the top mip across the window. --shots saves the
// window as PREFIX-MS.bmp that long after the upload, to see how a texture arrives.

#include <algorithm>
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

static const char* SHADER = R"(
cbuffer Constants : register(b0) { float2 offset; float scale; float tint; float2 uv_size; float brighten; float padding; };
Texture2D picture : register(t0);
SamplerState sampling : register(s0);
struct Out { float4 position : SV_Position; float2 uv : TEXCOORD; };
Out vs(float2 position : POSITION) {
    Out result;
    result.position = float4(position * scale + offset, 0, 1);
    result.uv = (position * 0.5 + 0.5) * uv_size;
    return result;
}
float4 ps(Out input) : SV_Target {
    float4 sampled = picture.Sample(sampling, input.uv);
    return float4(brighten > 0 ? pow(sampled.rgb, 1 / 2.2) : sampled.rgb * tint, 1);
}
)";

struct Constants {
    float offset[2];
    float scale;
    float tint;
    float uv_across = 1, uv_down = 1;
    // An sRGB texture reads as linear, and the window isn't sRGB
    float brighten = 0;
    float padding = 0;
};

// Saves what the window shows, as borrowed frames reach it by plain drawing
static void save_window(HWND window, int width, int height, const std::string& path) {
    HDC screen = GetDC(window);
    HDC memory = CreateCompatibleDC(screen);
    HBITMAP bitmap = CreateCompatibleBitmap(screen, width, height);
    SelectObject(memory, bitmap);
    BitBlt(memory, 0, 0, width, height, screen, 0, 0, SRCCOPY);
    BITMAPINFOHEADER info = {sizeof(info), width, height, 1, 24, BI_RGB};
    int pitch = ((width * 3 + 3) / 4) * 4;
    std::vector<char> pixels((size_t)pitch * height);
    GetDIBits(memory, bitmap, 0, height, pixels.data(), (BITMAPINFO*)&info, DIB_RGB_COLORS);
    BITMAPFILEHEADER head = {0x4d42, (DWORD)(sizeof(head) + sizeof(info) + pixels.size()), 0, 0, (DWORD)(sizeof(head) + sizeof(info))};
    FILE* file = fopen(path.c_str(), "wb");
    if (file) {
        fwrite(&head, sizeof(head), 1, file);
        fwrite(&info, sizeof(info), 1, file);
        fwrite(pixels.data(), 1, pixels.size(), file);
        fclose(file);
    }
    DeleteObject(bitmap);
    DeleteDC(memory);
    ReleaseDC(window, screen);
}

static void die(const char* what, HRESULT result) {
    printf("FAIL %s: 0x%08lx\n", what, (unsigned long)result);
    fflush(stdout);
    ExitProcess(1);
}

#define CHECK(call)                                       \
    do {                                                  \
        HRESULT check_result = (call);                    \
        if (FAILED(check_result)) die(#call, check_result); \
    } while (0)

static ID3DBlob* compile(const char* entry, const char* target) {
    ID3DBlob* code = nullptr;
    ID3DBlob* errors = nullptr;
    HRESULT result = D3DCompile(SHADER, strlen(SHADER), "probe", nullptr, nullptr, entry, target, 0, 0, &code, &errors);
    if (FAILED(result)) {
        if (errors) printf("%s\n", (const char*)errors->GetBufferPointer());
        die("D3DCompile", result);
    }
    return code;
}

int main(int argc, char** argv) {
    int frames = 600, draws = 2000, width = 1280, height = 720, upload_kib = 0;
    const char* upload_kind = "same";
    bool new_textures = false;
    int burst_frame = -1;
    const char* upload_file = nullptr;
    const char* shots_text = nullptr;
    int upload_side = 0;
    bool srgb = false, show = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--frames")) frames = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--draws")) draws = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--upload-kib")) upload_kib = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--upload-kind")) upload_kind = argv[i + 1];
        else if (!strcmp(argv[i], "--new-textures")) new_textures = atoi(argv[i + 1]) != 0;
        else if (!strcmp(argv[i], "--burst-frame")) burst_frame = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--upload-file")) upload_file = argv[i + 1];
        else if (!strcmp(argv[i], "--upload-side")) upload_side = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--srgb")) srgb = atoi(argv[i + 1]) != 0;
        else if (!strcmp(argv[i], "--show")) show = atoi(argv[i + 1]) != 0;
        else if (!strcmp(argv[i], "--shots")) shots_text = argv[i + 1];
        else if (!strcmp(argv[i], "--size")) sscanf(argv[i + 1], "%dx%d", &width, &height);
    }

    WNDCLASSW window_class = {};
    window_class.lpfnWndProc = DefWindowProcW;
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpszClassName = L"d3d11-probe";
    RegisterClassW(&window_class);
    RECT rect = {0, 0, width, height};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND window = CreateWindowExW(WS_EX_TOPMOST, L"d3d11-probe", L"d3d11-probe", WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, 0, 0,
                                  rect.right - rect.left, rect.bottom - rect.top, nullptr, nullptr, window_class.hInstance, nullptr);
    ShowWindow(window, SW_SHOWNOACTIVATE);

    DXGI_SWAP_CHAIN_DESC swap = {};
    swap.BufferDesc.Width = width;
    swap.BufferDesc.Height = height;
    swap.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swap.SampleDesc.Count = 1;
    swap.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap.BufferCount = 2;
    swap.OutputWindow = window;
    swap.Windowed = TRUE;
    swap.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    auto started = std::chrono::steady_clock::now();
    IDXGISwapChain* swapchain = nullptr;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    D3D_FEATURE_LEVEL level = {};
    CHECK(D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &swap, &swapchain, &device, &level, &context));

    IDXGIDevice* dxgi_device = nullptr;
    IDXGIAdapter* adapter = nullptr;
    CHECK(device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device));
    CHECK(dxgi_device->GetAdapter(&adapter));
    DXGI_ADAPTER_DESC description = {};
    CHECK(adapter->GetDesc(&description));
    printf("adapter      %ls, vendor %04x, %llu MiB, feature level %x\n", description.Description, description.VendorId,
           (unsigned long long)description.DedicatedVideoMemory >> 20, (unsigned)level);

    ID3D11Texture2D* back_buffer = nullptr;
    ID3D11RenderTargetView* target = nullptr;
    CHECK(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer));
    CHECK(device->CreateRenderTargetView(back_buffer, nullptr, &target));

    ID3DBlob* vertex_code = compile("vs", "vs_5_0");
    ID3DBlob* pixel_code = compile("ps", "ps_5_0");
    ID3D11VertexShader* vertex_shader = nullptr;
    ID3D11PixelShader* pixel_shader = nullptr;
    CHECK(device->CreateVertexShader(vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), nullptr, &vertex_shader));
    CHECK(device->CreatePixelShader(pixel_code->GetBufferPointer(), pixel_code->GetBufferSize(), nullptr, &pixel_shader));
    D3D11_INPUT_ELEMENT_DESC element = {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0};
    ID3D11InputLayout* layout = nullptr;
    CHECK(device->CreateInputLayout(&element, 1, vertex_code->GetBufferPointer(), vertex_code->GetBufferSize(), &layout));

    const float quad[] = {-1, -1, -1, 1, 1, -1, 1, 1};
    D3D11_BUFFER_DESC vertex_desc = {sizeof(quad), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER};
    D3D11_SUBRESOURCE_DATA vertex_data = {quad};
    ID3D11Buffer* vertices = nullptr;
    CHECK(device->CreateBuffer(&vertex_desc, &vertex_data, &vertices));

    D3D11_BUFFER_DESC constant_desc = {sizeof(Constants), D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE};
    ID3D11Buffer* constants = nullptr;
    CHECK(device->CreateBuffer(&constant_desc, nullptr, &constants));

    // A white texture, so a draw's colour is its tint; re-uploaded each frame with --upload-kib
    if (upload_file) {
        upload_kind = "bc7";
        upload_kib = 1;
    }
    const bool blocks = !strcmp(upload_kind, "bc7");
    // BC7 is a byte a texel, and a whole mip chain is 4/3 of its top mip
    int side = upload_kib > 0 ? (int)ceil(sqrt(upload_kib * 1024.0 / 4.0)) : 256;
    if (blocks) {
        side = 4;
        while (side * side * 4 / 3 < upload_kib * 1024) side *= 2;
        if (upload_file) side = upload_side;
    }
    std::vector<unsigned> texels(side * side, 0xffffffffu);
    // Each mip's offset in an upload, its row pitch and its rows
    struct Level { size_t offset; UINT pitch, rows; };
    std::vector<Level> levels;
    if (blocks) {
        size_t offset = 0;
        for (int mip = side;; mip /= 2) {
            UINT across = (UINT)((mip + 3) / 4);
            levels.push_back({offset, across * 16, across});
            offset += (size_t)across * across * 16;
            if (mip == 1) break;
        }
        texels.resize(offset / 4);
    } else {
        levels.push_back({0, (UINT)side * 4, (UINT)side});
    }
    // 16 frames' worth, which repeat: more than the compressor's window at the sizes that matter
    std::vector<std::vector<uint32_t>> uploads;
    if (upload_kib > 0 && strcmp(upload_kind, "same") != 0) {
        uploads.resize(upload_file ? 1 : upload_kib > 65536 ? 4 : 16);
        uint32_t noise = 0x2545f491u;
        for (size_t made = 0; made < uploads.size(); made++) {
            auto& upload = uploads[made];
            upload.resize(texels.size());
            for (size_t at = 0; at < upload.size(); at++) {
                if (!strcmp(upload_kind, "noise") || blocks) {
                    noise ^= noise << 13;
                    noise ^= noise >> 17;
                    noise ^= noise << 5;
                    upload[at] = noise;
                } else {
                    float value = 40.0f * sinf(at * 0.001f + made * 0.05f) + (at % 8) * 3.0f;
                    memcpy(&upload[at], &value, 4);
                }
            }
        }
    }
    if (upload_file) {
        FILE* file = fopen(upload_file, "rb");
        if (!file || fread(uploads[0].data(), 4, uploads[0].size(), file) != uploads[0].size()) {
            printf("FAIL %s isn't a chain of that size\n", upload_file);
            return 1;
        }
        fclose(file);
    }
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = side;
    texture_desc.Height = side;
    texture_desc.MipLevels = (UINT)levels.size();
    texture_desc.ArraySize = 1;
    texture_desc.Format = blocks ? (srgb ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM) : DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA texture_data = {texels.data(), (UINT)side * 4};
    ID3D11Texture2D* texture = nullptr;
    ID3D11ShaderResourceView* texture_view = nullptr;
    CHECK(device->CreateTexture2D(&texture_desc, blocks ? nullptr : &texture_data, &texture));
    CHECK(device->CreateShaderResourceView(texture, nullptr, &texture_view));

    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    ID3D11SamplerState* sampler = nullptr;
    CHECK(device->CreateSamplerState(&sampler_desc, &sampler));

    D3D11_VIEWPORT viewport = {0, 0, (float)width, (float)height, 0, 1};
    UINT stride = 8, zero = 0;
    double setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    printf("setup        %.0f ms\n", setup * 1000);
    fflush(stdout);

    started = std::chrono::steady_clock::now();
    const float background[4] = {0.1f, 0.1f, 0.3f, 1.0f};
    int last_upload = 0;
    // When to save the window, in milliseconds after the upload
    std::vector<int> shot_times;
    std::string shot_prefix;
    if (shots_text) {
        std::string text = shots_text;
        size_t comma = text.find(',');
        shot_prefix = text.substr(0, comma);
        while (comma != std::string::npos) {
            size_t next = text.find(',', comma + 1);
            shot_times.push_back(atoi(text.substr(comma + 1, next - comma - 1).c_str()));
            comma = next;
        }
    }
    size_t shots_taken = 0;
    std::chrono::steady_clock::time_point uploaded;
    double longest_frame = 0;
    for (int frame = 0; frame < frames; frame++) {
        MSG message;
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&message);

        // The back buffer's view is per buffer under flip, as a game's would be
        ID3D11Texture2D* current = nullptr;
        ID3D11RenderTargetView* current_target = nullptr;
        CHECK(swapchain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&current));
        CHECK(device->CreateRenderTargetView(current, nullptr, &current_target));
        context->OMSetRenderTargets(1, &current_target, nullptr);
        context->ClearRenderTargetView(current_target, background);
        context->RSSetViewports(1, &viewport);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        context->IASetInputLayout(layout);
        context->IASetVertexBuffers(0, 1, &vertices, &stride, &zero);
        context->VSSetShader(vertex_shader, nullptr, 0);
        context->PSSetShader(pixel_shader, nullptr, 0);
        context->VSSetConstantBuffers(0, 1, &constants);
        context->PSSetConstantBuffers(0, 1, &constants);
        context->PSSetShaderResources(0, 1, &texture_view);
        context->PSSetSamplers(0, 1, &sampler);
        auto frame_started = std::chrono::steady_clock::now();
        if (upload_kib > 0 && (burst_frame < 0 || frame == burst_frame)) {
            // A new texture each time is what loading is; the same one is what animating is
            if (new_textures && frame > 0) {
                texture_view->Release();
                texture->Release();
                CHECK(device->CreateTexture2D(&texture_desc, nullptr, &texture));
                CHECK(device->CreateShaderResourceView(texture, nullptr, &texture_view));
                context->PSSetShaderResources(0, 1, &texture_view);
            }
            if (uploads.empty()) texels[frame % texels.size()] = 0xfffffffeu - frame;
            // Made ahead, so that making them doesn't count against the frame
            last_upload = frame;
            uploaded = std::chrono::steady_clock::now();
            const char* from = (const char*)(uploads.empty() ? texels.data() : uploads[frame % uploads.size()].data());
            for (size_t mip = 0; mip < levels.size(); mip++) {
                context->UpdateSubresource(texture, (UINT)mip, nullptr, from + levels[mip].offset, levels[mip].pitch, 0);
            }
        }
        // Many small quads around the edge, then one at the centre that's checked on screen
        for (int draw = 0; draw <= draws; draw++) {
            Constants values;
            if (draw == draws) {
                values = {{0, 0}, show ? 1.0f : 0.2f, 1.0f};
                if (show) {
                    values.uv_across = (float)width / side;
                    values.uv_down = (float)height / side;
                    values.brighten = srgb ? 1.0f : 0.0f;
                }
            } else {
                float angle = (draw + frame) * 0.01f;
                values = {{0.8f * cosf(angle), 0.8f * sinf(angle)}, 0.02f, 0.25f + (draw % 4) * 0.25f};
            }
            D3D11_MAPPED_SUBRESOURCE mapped;
            CHECK(context->Map(constants, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));
            memcpy(mapped.pData, &values, sizeof(values));
            context->Unmap(constants, 0);
            context->Draw(4, 0);
        }
        CHECK(swapchain->Present(0, 0));
        if (shots_taken < shot_times.size() && frame >= last_upload && last_upload > 0 &&
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - uploaded).count() >= shot_times[shots_taken]) {
            save_window(window, width, height, shot_prefix + "-" + std::to_string(shot_times[shots_taken]) + ".bmp");
            shots_taken++;
        }
        // The first few are shaders being made, whatever's being measured
        if (frame >= 5) longest_frame = (std::max)(longest_frame, std::chrono::duration<double>(std::chrono::steady_clock::now() - frame_started).count());
        current_target->Release();
        current->Release();
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    printf("frames       %d of %dx%d, %d draws each, in %.2f s: %.1f frames/s\n", frames, width, height, draws + 1, seconds, frames / seconds);
    printf("longest      frame %.0f ms\n", longest_frame * 1000);
    fflush(stdout);

    // What the GPU holds must be what was last uploaded, byte for byte, however it got there
    if (upload_kib > 0 && burst_frame < frames) {
        D3D11_TEXTURE2D_DESC staging_desc = texture_desc;
        staging_desc.Usage = D3D11_USAGE_STAGING;
        staging_desc.BindFlags = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ID3D11Texture2D* staging = nullptr;
        CHECK(device->CreateTexture2D(&staging_desc, nullptr, &staging));
        context->CopyResource(staging, texture);
        const char* wanted = (const char*)(uploads.empty() ? texels.data() : uploads[last_upload % uploads.size()].data());
        for (size_t mip = 0; mip < levels.size(); mip++) {
            D3D11_MAPPED_SUBRESOURCE held;
            CHECK(context->Map(staging, (UINT)mip, D3D11_MAP_READ, 0, &held));
            // Which rows, and what's there instead, says what went wrong
            UINT first_bad = levels[mip].rows, bad_rows = 0, zero_rows = 0, other_upload = 0;
            for (UINT row = 0; row < levels[mip].rows; row++) {
                const char* row_held = (const char*)held.pData + (size_t)row * held.RowPitch;
                if (memcmp(row_held, wanted + levels[mip].offset + (size_t)row * levels[mip].pitch, levels[mip].pitch)) {
                    if (first_bad == levels[mip].rows) first_bad = row;
                    bad_rows++;
                    bool zeros = true;
                    for (UINT at = 0; at < levels[mip].pitch && zeros; at++) zeros = row_held[at] == 0;
                    if (zeros) zero_rows++;
                    for (auto& upload : uploads) {
                        if (!memcmp(row_held, (const char*)upload.data() + levels[mip].offset + (size_t)row * levels[mip].pitch, levels[mip].pitch)) { other_upload++; break; }
                    }
                }
            }
            if (bad_rows) {
                printf("FAIL mip %zu on the GPU isn't what was uploaded: %u of %u rows differ from row %u, %u all zeros, %u from another upload\n", mip, bad_rows, levels[mip].rows, first_bad, zero_rows, other_upload);
                return 1;
            }
            context->Unmap(staging, (UINT)mip);
        }
        staging->Release();
        printf("read back    %zu mips match what was uploaded\n", levels.size());
    }

    // Frames may still be on their way; the centre quad is white and everything else isn't
    Sleep(300);
    HDC dc = GetDC(window);
    // The window may be smaller than asked for, if the screen is
    RECT client = {};
    GetClientRect(window, &client);
    COLORREF centre = GetPixel(dc, client.right / 2, client.bottom / 2);
    COLORREF corner = GetPixel(dc, client.right / 2, client.bottom / 8);
    ReleaseDC(window, dc);
    int red = GetRValue(centre), green = GetGValue(centre), blue = GetBValue(centre);
    printf("on screen    centre (%d, %d, %d), above it (%d, %d, %d)\n", red, green, blue, GetRValue(corner), GetGValue(corner), GetBValue(corner));
    // With other uploads the texture isn't white, so the colours say nothing
    if (!strcmp(upload_kind, "same") && (red < 230 || green < 230 || blue < 230 || GetRValue(corner) > 60)) {
        printf("FAIL the window doesn't show the frame\n");
        return 1;
    }
    printf("PASS %ls: Direct3D 11 frames with shaders and a texture, on screen\n", description.Description);
    return 0;
}
