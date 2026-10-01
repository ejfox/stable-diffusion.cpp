#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "httplib.h"

#include "async_jobs.h"
#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "routes.h"
#include "runtime.h"

namespace fs = std::filesystem;

#ifdef HAVE_INDEX_HTML
#include "frontend/dist/gen_index_html.h"
#endif

static void print_usage(const char* argv0, const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv0 << " [options]\n\n";
    std::cout << "Svr Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nDefault Generation Options:\n";
    options_list[2].print();
}

static void parse_args(int argc,
                       const char** argv,
                       SDSvrParams& svr_params,
                       SDContextParams& ctx_params,
                       SDGenerationParams& default_gen_params) {
    std::vector<ArgOptions> options_vec = {
        svr_params.get_options(),
        ctx_params.get_options(),
        default_gen_params.get_options(),
    };

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argv[0], options_vec);
        exit(svr_params.normal_exit ? 0 : 1);
    }

    log_level = svr_params.log_level;
    log_color = svr_params.color;

    const bool random_seed_requested = default_gen_params.seed < 0;

    if (!svr_params.resolve_and_validate() ||
        !ctx_params.resolve_and_validate(IMG_GEN) ||
        !default_gen_params.resolve_and_validate(IMG_GEN,
                                                 ctx_params.lora_model_dir,
                                                 ctx_params.hires_upscalers_dir)) {
        print_usage(argv[0], options_vec);
        exit(1);
    }

    if (random_seed_requested) {
        default_gen_params.seed = -1;
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDSvrParams* svr_params = (SDSvrParams*)data;
    log_print(level, log, svr_params->log_level, svr_params->color);
}

static void preview_cb(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)step;
    (void)is_noisy;
    if (frame_count != 1 || image == nullptr) {
        return;
    }
    const SDSvrParams* svr_params = (const SDSvrParams*)data;
    const fs::path final_path     = svr_params->preview_path;
    const EncodedImageFormat fmt  = encoded_image_format_from_path(final_path.string());
    if (fmt == EncodedImageFormat::UNKNOWN) {
        LOG_ERROR("unsupported preview image extension: %s", final_path.string().c_str());
        return;
    }
    std::vector<uint8_t> bytes = encode_image_to_vector(fmt, image->data, image->width, image->height, image->channel);
    if (bytes.empty()) {
        LOG_ERROR("encode preview image failed");
        return;
    }
    // Write beside the target and rename, so clients polling the file never read a partial image.
    fs::path tmp_path = final_path;
    tmp_path += ".tmp";
    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
        if (!out) {
            LOG_ERROR("save preview image to '%s' failed", tmp_path.string().c_str());
            return;
        }
    }
    // On Windows the replace fails while a reader holds the target open; the next step retries.
    std::error_code ec;
    fs::rename(tmp_path, final_path, ec);
    if (ec) {
        LOG_WARN("move preview image to '%s' failed: %s", final_path.string().c_str(), ec.message().c_str());
    }
}

int main(int argc, const char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }
    SDSvrParams svr_params;
    SDContextParams ctx_params;
    SDGenerationParams default_gen_params;

    sd_set_log_callback(sd_log_cb, (void*)&svr_params);
    parse_args(argc, argv, svr_params, ctx_params, default_gen_params);

    LOG_VERBOSE("version: %s", version_string().c_str());
    LOG_VERBOSE("%s", sd_get_system_info());
    LOG_VERBOSE("%s", svr_params.to_string().c_str());
    LOG_VERBOSE("%s", ctx_params.to_string().c_str());
    LOG_VERBOSE("%s", default_gen_params.to_string().c_str());

    // With previews on, TAE (if given) is used only for previews and the full VAE decodes results.
    const bool previews             = !svr_params.preview_path.empty();
    const bool taesd_preview_only   = previews && !ctx_params.taesd_path.empty();
    sd_ctx_params_t sd_ctx_params   = ctx_params.to_sd_ctx_params_t(taesd_preview_only);
    SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));

    if (sd_ctx == nullptr) {
        LOG_ERROR("new_sd_ctx_t failed");
        return 1;
    }

    if (previews) {
        sd_set_preview_callback(preview_cb,
                                taesd_preview_only ? PREVIEW_TAE : PREVIEW_VAE,
                                svr_params.preview_interval,
                                true,
                                false,
                                (void*)&svr_params);
    }

    std::mutex sd_ctx_mutex;

    std::vector<LoraEntry> lora_cache;
    std::mutex lora_mutex;
    std::vector<UpscalerEntry> upscaler_cache;
    std::mutex upscaler_mutex;
    AsyncJobManager async_job_manager;
    ServerRuntime runtime = {
        sd_ctx.get(),
        &sd_ctx_mutex,
        &svr_params,
        &ctx_params,
        &default_gen_params,
        &lora_cache,
        &lora_mutex,
        &upscaler_cache,
        &upscaler_mutex,
        &async_job_manager,
    };

    std::thread async_worker(async_job_worker, std::ref(runtime));

    httplib::Server svr;

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string origin = req.get_header_value("Origin");
        if (origin.empty()) {
            origin = "*";
        }
        res.set_header("Access-Control-Allow-Origin", origin);
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::string index_html;
#ifdef HAVE_INDEX_HTML
    index_html.assign(reinterpret_cast<const char*>(index_html_bytes), index_html_size);
#else
    index_html = "Stable Diffusion Server is running";
#endif
    register_index_endpoints(svr, svr_params, index_html);
    register_openai_api_endpoints(svr, runtime);
    register_sdapi_endpoints(svr, runtime);
    register_sdcpp_api_endpoints(svr, runtime);

    LOG_INFO("listening on: http://%s:%d\n", svr_params.listen_ip.c_str(), svr_params.listen_port);
    svr.listen(svr_params.listen_ip, svr_params.listen_port);

    {
        std::lock_guard<std::mutex> lock(async_job_manager.mutex);
        async_job_manager.stop = true;
    }
    async_job_manager.cv.notify_all();
    async_worker.join();
    return 0;
}
