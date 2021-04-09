declare -arr sparsity=(0	0.05	0.1	0.15	0.2	0.25	0.3	0.35	0.4	0.45	0.5	0.55	0.6	0.65	0.7	0.75	0.8	0.85	0.9	0.95)
echo "t_qkv, t_O, t_L1, t_L2"
for s in ${sparsity[@]};
do
./test_algo_col 768 2304 128 $s
./test_algo_col 768 768 128 $s
./test_algo_col 768 3072 128 $s
./test_algo_col 3072 768 128 $s
echo ""
done