for dir in */pileup; do
    aws s3 sync "$dir" "s3://tf-binding-sites/pretraining/data/cell_lines/$dir"
done
