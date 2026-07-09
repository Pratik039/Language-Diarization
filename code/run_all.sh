#------------------------------------------------
#    Configuration
#------------------------------------------------

PYTHON=python 

SEG_DUR=1    # maximum segment duration in sec
SEG_SHIFT=0.2 # shift to the next segment in sec
SEG_OVRLAP=`echo "$SEG_DUR - $SEG_SHIFT" | bc -l` 


#------------------------------------------------
#    Data directories
#------------------------------------------------

DISPLACE_DATAPATH="../data/"
DATA="dev"
DISPLACE_DATA="$DISPLACE_DATAPATH/$DATA"

ref_RTTM="$DISPLACE_DATA/${DATA}_original.rttm"

OUTPUT_DIR="../output"


#------------------------------------------------
#    VAD and Data Preparation
#------------------------------------------------

VAD_SUBDIR="$DATA/"
SEG_SUBDIR="${DATA}seg/"
KAL_SUBDIR="kal_${DATA}seg/"

$PYTHON pyannote_vad.py $DISPLACE_DATA  $OUTPUT_DIR $VAD_SUBDIR $SEG_SUBDIR

path_segments="$OUTPUT_DIR/$SEG_SUBDIR"
path_new_kaldi_segs="$OUTPUT_DIR/$KAL_SUBDIR"
mkdir -p $path_new_kaldi_segs
for path in $(ls $path_segments)
do
    g=${path_segments}/${path}
    $PYTHON seg2kaldi.py $g ${path_new_kaldi_segs}${path}.txt
done

SUBSEG_SUBDIR="${DATA}_subseg/"
path_subsegs="$OUTPUT_DIR/$SUBSEG_SUBDIR"
mkdir -p $path_subsegs

$PYTHON create_subseg.py --input_folder $path_new_kaldi_segs --out_subsegments_folder $path_subsegs --max-segment-duration $SEG_DUR --overlap-duration $SEG_OVRLAP --max-remaining-duration $SEG_DUR --constant-duration False

#-------------------------------------------------
#    Feature Extraction
#-------------------------------------------------

#
FEAT_SUBDIR="${DATA}_feat/"
mkdir -p $OUTPUT_DIR/$FEAT_SUBDIR 
$PYTHON feat_extr_whisper.py --aud_path $DISPLACE_DATA  --out_dir $OUTPUT_DIR/$FEAT_SUBDIR --seg_dir  $OUTPUT_DIR/$SUBSEG_SUBDIR



#-------------------------------------------------
#    Clustering (AHC)
#-------------------------------------------------

FEAT_SEG_TSV="${DATA}_whisper_feat.tsv"
RTTM_OUT_DIR="${DATA}_rttm_outdir"
CLUSTER_ALGO="AHC"
mkdir -p $OUTPUT_DIR/$RTTM_OUT_DIR

ls -1 $OUTPUT_DIR/$SUBSEG_SUBDIR/*.txt  | sort > $OUTPUT_DIR/temp_segs
ls -1 $OUTPUT_DIR/$FEAT_SUBDIR/*.npy | sort > $OUTPUT_DIR/temp_posteriors
paste $OUTPUT_DIR/temp_posteriors $OUTPUT_DIR/temp_segs > $OUTPUT_DIR/$FEAT_SEG_TSV
$PYTHON clustering.py $OUTPUT_DIR/$FEAT_SEG_TSV $OUTPUT_DIR/$RTTM_OUT_DIR $CLUSTER_ALGO
	

#-------------------------------------------------
#    Scoring
#-------------------------------------------------

sys_RTTM=$OUTPUT_DIR/$RTTM_OUT_DIR/sys_whisper.rttm

cat $OUTPUT_DIR/$RTTM_OUT_DIR/*.rttm  > $sys_RTTM
$PYTHON score.py -r $ref_RTTM -s $sys_RTTM 
