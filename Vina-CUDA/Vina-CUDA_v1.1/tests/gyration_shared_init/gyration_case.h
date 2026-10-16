// Host-side part of the gyration_radius_update() test: input sets, CPU reference,
// a model of the block-wide reduction, and result reporting. No CUDA API is used here.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "kernel2.h" // flo, atom_cl, m_coords_cl, MAX_NUM_OF_ATOMS, EL_TYPE_H

namespace gyration_test {

constexpr int kThreads = 32; // threads per block in kernel2_update (block_size 16 x 2)
constexpr int kSizes[] = {16, 31, 32, 55, 127};
constexpr int kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
constexpr int kHeavyElType = 1; // EL_TYPE_C; any type != EL_TYPE_H is a heavy atom
constexpr double kTolerance = 1e-5;

// All coordinates are origin +/- d along one axis, so every distance and every
// partial sum is exactly representable in float.
constexpr flo kOrigin[3] = {1.5f, -2.25f, 0.75f};
constexpr flo kMeasureHeavyDist = 1.0f;    // reference R_g of every measurement set is exactly 1
constexpr flo kMeasureHydrogenDist = 5.0f; // changes R_g if hydrogens were counted
constexpr flo kUnusedDist = 7.0f;          // atoms with index >= n: changes R_g if they were read
constexpr flo kPrefillDist = 3.0f;         // leftover tail has acc / counter = 9 != 1 = R_g^2

enum class Pattern { Zero, Nonzero };

inline const char *pattern_name(Pattern p) { return p == Pattern::Zero ? "zero" : "nonzero"; }

struct AtomSet
{
	atom_cl atoms[MAX_NUM_OF_ATOMS];
	m_coords_cl coords;
};

inline void place_atom(AtomSet &s, int i, int el_type, flo dist)
{
	static const flo dirs[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
	std::memset(&s.atoms[i], 0, sizeof(atom_cl));
	s.atoms[i].types[0] = el_type;
	for (int j = 0; j < 3; j++)
	{
		s.coords.coords[i][j] = kOrigin[j] + dist * dirs[i % 6][j];
		s.atoms[i].coords[j] = s.coords.coords[i][j];
	}
}

// Ligand of n atoms: every 4th atom is a hydrogen; atoms past n are decoys.
inline void make_measure_set(AtomSet &s, int n)
{
	for (int i = 0; i < MAX_NUM_OF_ATOMS; i++)
	{
		if (i >= n)
			place_atom(s, i, kHeavyElType, kUnusedDist);
		else if (i % 4 == 3)
			place_atom(s, i, EL_TYPE_H, kMeasureHydrogenDist);
		else
			place_atom(s, i, kHeavyElType, kMeasureHeavyDist);
	}
}

// Input of the prefill call (m_lig_end = MAX_NUM_OF_ATOMS):
// Zero    - all hydrogens, leaves acc = 0 and counter = 0 in every entry;
// Nonzero - all heavy atoms at distance 3, leaves entries with acc / counter = 9.
inline void make_prefill_set(AtomSet &s, Pattern p)
{
	for (int i = 0; i < MAX_NUM_OF_ATOMS; i++)
		place_atom(s, i, p == Pattern::Zero ? EL_TYPE_H : kHeavyElType, kPrefillDist);
}

inline double reference_rg(const AtomSet &s, int n)
{
	double acc = 0;
	int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (s.atoms[i].types[0] == EL_TYPE_H)
			continue;
		for (int j = 0; j < 3; j++)
		{
			const double d = double(s.coords.coords[i][j]) - double(kOrigin[j]);
			acc += d * d;
		}
		++count;
	}
	return count > 0 ? std::sqrt(acc / count) : 0.0;
}

// The __shared__ arrays of gyration_radius_update(), carried between calls.
struct SharedState
{
	float acc[MAX_NUM_OF_ATOMS];
	int counter[MAX_NUM_OF_ATOMS];
};

// Sequential model of gyration_radius_update() for kThreads threads (MAX_NUM_OF_ATOMS > kThreads
// branch). zero_limit = n reproduces the original code, zero_limit = MAX_NUM_OF_ATOMS the fix.
inline flo model_call(SharedState &sh, const AtomSet &s, int n, int zero_limit)
{
	for (int i = 0; i < zero_limit; i++)
	{
		sh.counter[i] = 0;
		sh.acc[i] = 0.0f;
	}
	for (int i = 0; i < n; i++)
	{
		if (s.atoms[i].types[0] == EL_TYPE_H)
			continue;
		const flo dx = s.coords.coords[i][0] - kOrigin[0];
		const flo dy = s.coords.coords[i][1] - kOrigin[1];
		const flo dz = s.coords.coords[i][2] - kOrigin[2];
		sh.acc[i] = dx * dx + dy * dy + dz * dz;
		sh.counter[i] = 1;
	}
	const int tmp = MAX_NUM_OF_ATOMS / kThreads;
	for (int t = 0; t < kThreads; t++)
	{
		for (int i = 1; i < tmp; i++)
		{
			sh.acc[t] += sh.acc[t + kThreads * i];
			sh.counter[t] += sh.counter[t + kThreads * i];
		}
		if (t < MAX_NUM_OF_ATOMS % kThreads)
		{
			sh.acc[t] += sh.acc[t + tmp * kThreads];
			sh.counter[t] += sh.counter[t + tmp * kThreads];
		}
	}
	for (int off = kThreads >> 1; off > 0; off >>= 1)
	{
		for (int t = 0; t < off; t++)
		{
			sh.acc[t] += sh.acc[t + off];
			sh.counter[t] += sh.counter[t + off];
		}
	}
	return (sh.counter[0] > 0) ? std::sqrt(sh.acc[0] / sh.counter[0]) : 0;
}

// Value the original code must return after the given prefill.
inline flo original_model_rg(const AtomSet &prefill, const AtomSet &measure, int n)
{
	SharedState sh = {};
	model_call(sh, prefill, MAX_NUM_OF_ATOMS, MAX_NUM_OF_ATOMS);
	return model_call(sh, measure, n, n);
}

inline bool close_to(double got, double expected)
{
	return std::fabs(got - expected) <= kTolerance * std::max(1.0, std::fabs(expected));
}

struct CaseResult
{
	Pattern pattern;
	int n;
	double ref;
	flo got;        // value returned by thread 0
	flo orig_model; // prediction for the original code
	bool finite;        // all threads returned a finite value
	bool threads_agree; // all threads returned the same bits
};

// Prints one machine-readable line; returns true if the GPU result matches the reference.
inline bool report(const CaseResult &r)
{
	const bool vs_ref = r.finite && close_to(r.got, r.ref);
	const bool vs_orig = r.finite && close_to(r.got, r.orig_model);
	std::printf("RESULT mode=prefill pattern=%s n=%d ref=%.7g got=%.7g orig_model=%.7g "
				"vs_ref=%s vs_orig_model=%s finite=%s threads_agree=%s\n",
				pattern_name(r.pattern), r.n, r.ref, double(r.got), double(r.orig_model),
				vs_ref ? "MATCH" : "MISMATCH", vs_orig ? "MATCH" : "MISMATCH",
				r.finite ? "yes" : "no", r.threads_agree ? "yes" : "no");
	return vs_ref && r.threads_agree;
}

} // namespace gyration_test
