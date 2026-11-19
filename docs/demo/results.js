const results = {
  "4-balanced": {
    "serial": {
      "output_tok_s": 49.03140629455743,
      "p95_ttft_s": 0.08845597691833973,
      "peak_admitted": 4.0
    },
    "batched": {
      "output_tok_s": 108.37972936401778,
      "p95_ttft_s": 0.08871829323470592,
      "peak_admitted": 4.0,
      "mean_decode_width": 2.4242424242424243
    },
    "speedup": 2.2104144578869263
  },
  "16-balanced": {
    "serial": {
      "output_tok_s": 49.42050267108415,
      "p95_ttft_s": 4.74558144621551,
      "peak_admitted": 9.0
    },
    "batched": {
      "output_tok_s": 194.4501395930237,
      "p95_ttft_s": 1.0499574784189463,
      "peak_admitted": 9.0,
      "mean_decode_width": 4.776119402985074
    },
    "speedup": 3.934604649555621
  },
  "32-balanced": {
    "serial": {
      "output_tok_s": 49.45331556915211,
      "p95_ttft_s": 10.65524299070239,
      "peak_admitted": 9.0
    },
    "batched": {
      "output_tok_s": 214.99392239646252,
      "p95_ttft_s": 2.2590077817440033,
      "peak_admitted": 9.0,
      "mean_decode_width": 5.47008547008547
    },
    "speedup": 4.347411693677643
  },
  "16-short-heavy": {
    "serial": {
      "output_tok_s": 49.47994554495815,
      "p95_ttft_s": 2.0957792941480875,
      "peak_admitted": 12.0
    },
    "batched": {
      "output_tok_s": 194.00752337562375,
      "p95_ttft_s": 0.5192536916583776,
      "peak_admitted": 12.0,
      "mean_decode_width": 5.2682926829268295
    },
    "speedup": 3.9209324351286097
  },
  "16-long-heavy": {
    "serial": {
      "output_tok_s": 49.54394199484221,
      "p95_ttft_s": 5.545169534161687,
      "peak_admitted": 7.0
    },
    "batched": {
      "output_tok_s": 231.3876772413945,
      "p95_ttft_s": 1.2221018988639116,
      "peak_admitted": 7.0,
      "mean_decode_width": 5.653333333333333
    },
    "speedup": 4.670352578433973
  }
};
