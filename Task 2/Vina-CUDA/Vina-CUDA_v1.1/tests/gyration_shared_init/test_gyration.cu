// Isolated test for gyration_radius_update() from inc/cuda/mutate_conf.cuh.
//
// The function reduces all MAX_NUM_OF_ATOMS entries of its __shared__ arrays acc/counter,
// while the original code zeroes only [0, m_lig_end). These function-scope __shared__ arrays
// persist between calls within a block, so a first call with m_lig_end = MAX_NUM_OF_ATOMS
// ("prefill") writes a known pattern into the whole arrays, and a second call ("measure")
// shows whether the result depends on what is left in the tail.
//
//   --mode prefill  (default) zero and nonzero tail for every size in kSizes;
//                   exit 0 if every result matches the CPU reference, 1 otherwise
//   --mode raw      one call per launch without prefill, results not checked;
//                   meant for compute-sanitizer --tool initcheck
//
// Exit codes: 0 match, 1 mismatch, 2 CUDA or usage error, 77 no CUDA device.
//
// Build against the header under test, e.g. from Vina-CUDA_v1.1:
//   nvcc -O2 -lineinfo -DLARGE_BOX -I inc/cuda -I inc tests/gyration_shared_init/test_gyration.cu -o test_gyration

#include <cstdlib>
#include <cstring>
#include <vector>

#include "mutate_conf.cuh"
#include "gyration_case.h"

using namespace gyration_test;

namespace {

constexpr int kExitMatch = 0;
constexpr int kExitMismatch = 1;
constexpr int kExitError = 2;
constexpr int kExitSkip = 77;

#define CUDA_TRY(call)                                                                       \
	do                                                                                       \
	{                                                                                        \
		cudaError_t err_ = (call);                                                           \
		if (err_ != cudaSuccess)                                                             \
		{                                                                                    \
			std::fprintf(stderr, "ERROR: %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, \
						 __LINE__, #call);                                                   \
			std::exit(kExitError);                                                           \
		}                                                                                    \
	} while (0)

} // namespace

// One block of kThreads threads, as one docking lane of kernel2_update.
// prefill and measure are separate buffers prepared on the host, so the only barrier needed
// is after the prefill call: every thread must finish reading acc[0]/counter[0] in its return
// statement before the measure call starts zeroing them.
__global__ void gyration_probe(AtomSet *prefill, AtomSet *measure, int n, const flo *origin, flo *out)
{
	const int tid = threadIdx.x;
	const int tpb = blockDim.x;
	if (prefill != nullptr)
	{
		(void)gyration_radius_update(0, MAX_NUM_OF_ATOMS, prefill->atoms, &prefill->coords, origin, tid, tpb);
		__syncthreads();
	}
	out[tid] = gyration_radius_update(0, n, measure->atoms, &measure->coords, origin, tid, tpb);
}

static std::vector<flo> run_block(const AtomSet *prefill, const AtomSet &measure, int n)
{
	AtomSet *d_prefill = nullptr;
	AtomSet *d_measure = nullptr;
	flo *d_origin = nullptr;
	flo *d_out = nullptr;

	if (prefill != nullptr)
	{
		CUDA_TRY(cudaMalloc(&d_prefill, sizeof(AtomSet)));
		CUDA_TRY(cudaMemcpy(d_prefill, prefill, sizeof(AtomSet), cudaMemcpyHostToDevice));
	}
	CUDA_TRY(cudaMalloc(&d_measure, sizeof(AtomSet)));
	CUDA_TRY(cudaMemcpy(d_measure, &measure, sizeof(AtomSet), cudaMemcpyHostToDevice));
	CUDA_TRY(cudaMalloc(&d_origin, sizeof(kOrigin)));
	CUDA_TRY(cudaMemcpy(d_origin, kOrigin, sizeof(kOrigin), cudaMemcpyHostToDevice));
	CUDA_TRY(cudaMalloc(&d_out, kThreads * sizeof(flo)));
	CUDA_TRY(cudaMemset(d_out, 0xFF, kThreads * sizeof(flo))); // NaN: a missing write is not finite

	gyration_probe<<<1, kThreads>>>(d_prefill, d_measure, n, d_origin, d_out);
	CUDA_TRY(cudaGetLastError());
	CUDA_TRY(cudaDeviceSynchronize());

	std::vector<flo> out(kThreads);
	CUDA_TRY(cudaMemcpy(out.data(), d_out, kThreads * sizeof(flo), cudaMemcpyDeviceToHost));

	CUDA_TRY(cudaFree(d_out));
	CUDA_TRY(cudaFree(d_origin));
	CUDA_TRY(cudaFree(d_measure));
	if (d_prefill != nullptr)
		CUDA_TRY(cudaFree(d_prefill));
	return out;
}

static bool all_finite(const std::vector<flo> &v)
{
	for (flo x : v)
		if (!std::isfinite(x))
			return false;
	return true;
}

static bool all_same_bits(const std::vector<flo> &v)
{
	for (flo x : v)
		if (std::memcmp(&x, &v[0], sizeof(flo)) != 0)
			return false;
	return true;
}

static int run_prefill_mode()
{
	static AtomSet prefill;
	static AtomSet measure;
	bool all_match = true;
	for (Pattern p : {Pattern::Zero, Pattern::Nonzero})
	{
		make_prefill_set(prefill, p);
		for (int n : kSizes)
		{
			make_measure_set(measure, n);
			const std::vector<flo> out = run_block(&prefill, measure, n);
			CaseResult r;
			r.pattern = p;
			r.n = n;
			r.ref = reference_rg(measure, n);
			r.got = out[0];
			r.orig_model = original_model_rg(prefill, measure, n);
			r.finite = all_finite(out);
			r.threads_agree = all_same_bits(out);
			all_match = report(r) && all_match;
		}
	}
	return all_match ? kExitMatch : kExitMismatch;
}

static int run_raw_mode()
{
	static AtomSet measure;
	for (int n : kSizes)
	{
		make_measure_set(measure, n);
		const std::vector<flo> out = run_block(nullptr, measure, n);
		std::printf("RAW n=%d ref=%.7g got=%.7g\n", n, reference_rg(measure, n), double(out[0]));
	}
	return kExitMatch;
}

int main(int argc, char **argv)
{
	const char *mode = "prefill";
	if (argc == 3 && std::strcmp(argv[1], "--mode") == 0)
		mode = argv[2];
	else if (argc != 1)
	{
		std::fprintf(stderr, "usage: %s [--mode prefill|raw]\n", argv[0]);
		return kExitError;
	}
	if (std::strcmp(mode, "prefill") != 0 && std::strcmp(mode, "raw") != 0)
	{
		std::fprintf(stderr, "ERROR: unknown mode '%s'\n", mode);
		return kExitError;
	}

	int device_count = 0;
	const cudaError_t err = cudaGetDeviceCount(&device_count);
	if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver || (err == cudaSuccess && device_count == 0))
	{
		std::printf("SKIP: no usable CUDA device (%s)\n", cudaGetErrorName(err));
		return kExitSkip;
	}
	CUDA_TRY(err);

	cudaDeviceProp prop;
	int runtime_version = 0;
	int driver_version = 0;
	CUDA_TRY(cudaGetDeviceProperties(&prop, 0));
	CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
	CUDA_TRY(cudaDriverGetVersion(&driver_version));
	std::printf("DEVICE name=\"%s\" pci=%08x:%02x:%02x.0 cc=%d.%d runtime=%d driver=%d\n", prop.name,
				prop.pciDomainID, prop.pciBusID, prop.pciDeviceID, prop.major, prop.minor, runtime_version,
				driver_version);

	return std::strcmp(mode, "raw") == 0 ? run_raw_mode() : run_prefill_mode();
}
