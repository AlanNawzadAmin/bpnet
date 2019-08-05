"""
Run modisco
"""
import logging
import matplotlib.pyplot as plt
from argh.decorators import named, arg
import shutil
import pandas as pd
import os
from collections import OrderedDict
from tqdm import tqdm
from pathlib import Path
from bpnet.utils import write_pkl, render_ipynb, remove_exists, add_file_logging, create_tf_session
from bpnet.cli.contrib import ContribFile
from bpnet.cli.train import _get_gin_files, log_gin_config
# ContribFile
from bpnet.modisco.results import ModiscoResult
from concise.utils.helper import write_json, read_json
import gin
import h5py
import numpy as np
import inspect
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------
# load functions for the modisco directory


def load_included_samples(modisco_dir):
    return np.load(os.path.join(modisco_dir, "modisco-run.subset-contrib-file.npz"))


def load_ranges(modisco_dir):
    modisco_dir = Path(modisco_dir)
    included_samples = load_included_samples(modisco_dir)

    kwargs = read_json(modisco_dir / "modisco-run.kwargs.json")
    d = ContribFile(kwargs["contrib_file"], included_samples)
    df = d.get_ranges()
    d.close()
    return df


def get_nonredundant_example_idx(ranges, width=200):
    """Get non - overlapping intervals(in the central region)

    Args:
      ranges: pandas.DataFrame returned by bpnet.cli.modisco.load_ranges
      width: central region considered that should not overlap between
         any interval
    """
    from pybedtools import BedTool
    from bpnet.preproc import resize_interval
    # 1. resize ranges
    ranges['example_idx'] = np.arange(len(ranges))  # make sure
    r = ranges[['chrom', 'start', 'end', 'example_idx']]  # add also the strand information
    r = resize_interval(r, width, ignore_strand=True)

    bt = BedTool.from_dataframe(r)
    btm = bt.sort().merge()
    df = btm.to_dataframe()
    df = df[(df.end - df.start) < width * 2]

    r_overlaps = bt.intersect(BedTool.from_dataframe(df), wb=True).to_dataframe()
    keep_idx = r_overlaps.drop_duplicates(['score', 'strand', 'thickStart'])['name'].astype(int)

    return keep_idx


# --------------------------------------------
@gin.configurable
def modisco_run(output_path,  # specified by bpnet_modisco_run
                task_names,
                contrib_scores,
                hypothetical_contribs,
                one_hot,
                null_per_pos_scores,
                # specified by gin-config
                workflow=gin.REQUIRED,  # TfModiscoWorkflow
                report=None):  # reports to use
    """
    Args:
      workflow: TfModiscoWorkflow objects
      report: path to the report ipynb
    """
    modisco_results = workflow(task_names=task_names,
                               contrib_scores=contrib_scores,
                               hypothetical_contribs=hypothetical_contribs,
                               one_hot=one_hot,
                               null_per_pos_scores=null_per_pos_scores)
    # save the results
    logger.info(f"Saving modisco file to {output_path}")
    grp = h5py.File(output_path)
    modisco_results.save_hdf5(grp)
    grp.flush()
    grp.close()

    if report is not None:
        if report is not None:
            report = os.path.abspath(os.path.expanduser(report))
            if not os.path.exists(report):
                raise ValueError(f"Report file {report} doesn't exist")

        logger.info("Running the report")
        # Run the jupyter notebook
        render_ipynb(report,
                     os.path.join(os.path.dirname(output_path), os.path.basename(report)),
                     params=dict(modisco_file=output_path,
                                 modisco_dir=os.path.dirname(output_path)))


@named("modisco-run")
@arg('contrib_file',
     help='path to the hdf5 file containing contribution scores')
@arg('output_dir',
     help='output file directory')
@arg('--null-contrib-file',
     help='Path to the null contribution scores')
@arg('--overwrite',
     help='if True, the output directory will be overwritten')
@arg('--premade',
     help='pre-made config file specifying modisco hyper-paramters to use.')
@arg('--config',
     help='gin config file path(s) specifying the modisco workflow parameters.'
     ' Parameters specified here override the --premade parameters. Multiple '
     'config files can be separated by comma separation (i.e. --config=file1.gin,file2.gin)')
@arg('--override',
     help='semi-colon separated list of additional gin bindings to use')
@arg("--contrib-wildcard",
     help="Wildcard of the contribution scores to use for running modisco. For example, */profile/wn computes"
     "uses the profile contribution scores for all the tasks (*) using the wn normalization (see bpnet.heads.py)."
     "*/counts/pre-act uses the total count contribution scores for all tasks w.r.t. the pre-activation output "
     "of prediction heads. Multiple wildcards can be by comma-separating them.")
@arg('--only-task-regions',
     help='If specified, only the contribution scores from regions corresponding to the tasks specified '
     'in --contrib-wildcard will be used. For example, if dataspec.yml contained Oct4 and Sox2 peaks when '
     'generating the contrib_file and `--contrib-wildcard=Oct4/profile/wn`, then modisco will be only ran '
     'in the Oct4 peaks. If `--contrib-wildcard=Oct4/profile/wn,Sox2/profile/wn` or `--contrib-wildcard=*/profile/wn`, '
     'then peaks of both Sox2 and Oct4 will be used.')
@arg('--filter-npy',
     help='File path to the .npz file containing a boolean one-dimensional numpy array of the same length'
     'as the contrib_file. Modisco will be ran on a subset of regions in the contrib_file '
     'where this array has value=True.')
@arg('--exclude-chr',
     help='Comma-separated list of chromosomes to exclude.')
@arg('--num-workers',
     help='number of workers to use in parallel for running modisco')
@arg('--gpu',
     help='which gpu to use. Example: gpu=1')
@arg('--memfrac-gpu',
     help='what fraction of the GPU memory to use')
@arg('--overwrite',
     help='If True, the output files will be overwritten if they already exist.')
def bpnet_modisco_run(contrib_file,
                      output_dir,
                      null_contrib_file=None,
                      premade='modisco-50k',
                      config=None,
                      override='',
                      contrib_wildcard="*/profile/wn",  # on which contribution scores to run modisco
                      only_task_regions=False,
                      filter_npy=None,
                      exclude_chr="",
                      num_workers=10,
                      gpu=None,  # no need to use a gpu by default
                      memfrac_gpu=0.45,
                      overwrite=False,
                      ):
    """Run TF-MoDISco on the contribution scores stored in the contribution score file
    generated by `bpnet contrib`.
    """
    add_file_logging(output_dir, logger, 'modisco-run')
    if gpu is not None:
        logger.info(f"Using gpu: {gpu}, memory fraction: {memfrac_gpu}")
        create_tf_session(gpu, per_process_gpu_memory_fraction=memfrac_gpu)
    else:
        # Don't use any GPU's
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

    import modisco
    assert '/' in contrib_wildcard

    if filter_npy is not None:
        filter_npy = os.path.abspath(filter_npy)
    if config is not None:
        config = os.path.abspath(config)

    # setup output file paths
    output_path = os.path.abspath(os.path.join(output_dir, "modisco.h5"))
    remove_exists(output_path, overwrite=overwrite)
    output_filter_npy = os.path.abspath(os.path.join(output_dir, 'modisco-run.subset-contrib-file.npy'))
    remove_exists(output_filter_npy, overwrite=overwrite)
    kwargs_json_file = os.path.join(output_dir, "modisco-run.kwargs.json")
    remove_exists(kwargs_json_file, overwrite=overwrite)
    if config is not None:
        config_output_file = os.path.join(output_dir, 'modisco-run.input-config.gin')
        remove_exists(config_output_file, overwrite=overwrite)
        shutil.copyfile(config, config_output_file)

    # save the hyper-parameters
    write_json(dict(contrib_file=os.path.abspath(contrib_file),
                    output_dir=str(output_dir),
                    null_contrib_file=null_contrib_file,
                    config=str(config),
                    override=override,
                    contrib_wildcard=contrib_wildcard,
                    only_task_regions=only_task_regions,
                    filter_npy=str(filter_npy),
                    exclude_chr=exclude_chr,
                    num_workers=num_workers,
                    overwrite=overwrite,
                    output_filter_npy=output_filter_npy,
                    gpu=gpu,
                    memfrac_gpu=memfrac_gpu),
               kwargs_json_file)

    # setup the gin config using premade, config and override
    cli_bindings = [f'num_workers={num_workers}']
    gin.parse_config_files_and_bindings(_get_gin_files(premade, config),
                                        bindings=cli_bindings + override.split(";"),
                                        # NOTE: custom files were inserted right after
                                        # ther user's config file and before the `override`
                                        # parameters specified at the command-line
                                        skip_unknown=False)
    log_gin_config(output_dir, prefix='modisco-run.')
    # --------------------------------------------

    # load the contribution file
    logger.info(f"Loading the contribution file: {contrib_file}")
    cf = ContribFile(contrib_file)
    tasks = cf.get_tasks()

    # figure out subset_tasks
    subset_tasks = set()
    for w in contrib_wildcard.split(","):
        task, head, head_summary = w.split("/")
        if task == '*':
            subset_tasks = None
        else:
            if task not in tasks:
                raise ValueError(f"task {task} not found in tasks: {tasks}")
            subset_tasks.add(task)
    if subset_tasks is not None:
        subset_tasks = list(subset_tasks)

    # --------------------------------------------
    # subset the intervals
    logger.info(f"Loading ranges")
    ranges = cf.get_ranges()
    # include all samples at the beginning
    include_samples = np.ones(len(cf)).astype(bool)

    # --only-task-regions
    if only_task_regions:
        if subset_tasks is None:
            logger.warn("contrib_wildcard contains all tasks (specified by */<head>/<summary>). Not using --only-task-regions")
        elif np.all(ranges['interval_from_task'] == ''):
            raise ValueError("Contribution file wasn't created from multiple set of peaks. "
                             "E.g. interval_from_task='' for all ranges. Please disable --only-task-regions")
        else:
            logger.info(f"Subsetting ranges according to `interval_from_task`")
            include_samples = include_samples & ranges['interval_from_task'].isin(subset_tasks).values
            logger.info(f"Using {include_samples.sum()} / {len(include_samples)} regions after --only-task-regions subset")

    # --exclude-chr
    if exclude_chr:
        logger.info(f"Excluding chromosomes: {exclude_chr}")
        chromosomes = ranges['chr']
        include_samples = include_samples & (~pd.Series(chromosomes).isin(exclude_chr)).values
        logger.info(f"Using {include_samples.sum()} / {len(include_samples)} regions after --exclude-chr subset")

    # -- filter-npy
    if filter_npy is not None:
        print(f"Loading a filter file from {filter_npy}")
        include_samples = include_samples & np.load(filter_npy)
        logger.info(f"Using {include_samples.sum()} / {len(include_samples)} regions after --filter-npy subset")

    # store the subset-contrib-file.npy
    logger.info(f"Saving the included samples from ContribFile to {output_filter_npy}")
    np.save(output_filter_npy, include_samples)
    # --------------------------------------------

    # convert to indices
    idx = np.arange(len(include_samples))[include_samples]
    seqs = cf.get_seq(idx=idx)

    # fetch the contribution scores from the importance score file
    # expand * to use all possible values
    # TODO - allow this to be done also for all the heads?
    hyp_contrib = {}
    task_names = []
    for w in contrib_wildcard.split(","):
        wc_task, head, head_summary = w.split("/")
        if task == '*':
            use_tasks = tasks
        else:
            use_tasks = [wc_task]
        for task in use_tasks:
            key = f"{task}/{head}/{head_summary}"
            task_names.append(key)
            hyp_contrib[key] = cf._subset(cf.data[f'/hyp_contrib/{key}'], idx=idx)
    contrib = {k: v * seqs for k, v in hyp_contrib.items()}

    if null_contrib_file is not None:
        logger.info(f"Using null-contrib-file: {null_contrib_file}")
        null_cf = ContribFile(null_contrib_file)
        null_seqs = null_cf.get_seq()
        null_per_pos_scores = {key: null_seqs * null_cf.data[f'/hyp_contrib/{key}'][:]
                               for key in task_names}
    else:
        # default Null distribution. Requires modisco 5.0
        logger.info(f"Using default null_contrib_scores")
        null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=10000)

    # run modisco.
    # NOTE: `workflow` and `report` parameters are provided by gin config files
    modisco_run(task_names=task_names,
                output_path=output_path,
                contrib_scores=contrib,
                hypothetical_contribs=hyp_contrib,
                one_hot=seqs,
                null_per_pos_scores=null_per_pos_scores)


def modisco_plot(modisco_dir,
                 output_dir,
                 # filter_npy=None,
                 # ignore_dist_filter=False,
                 figsize=(10, 10), contribsf=None):
    """Plot the results of a modisco run

    Args:
      modisco_dir: modisco directory
      output_dir: Output directory for writing the results
      figsize: Output figure size
      contribsf: [optional] modisco contribution score file (ContribFile)
    """
    plt.switch_backend('agg')
    add_file_logging(output_dir, logger, 'modisco-plot')
    from bpnet.plot.vdom import write_heatmap_pngs
    from bpnet.plot.profiles import plot_profiles
    from bpnet.utils import flatten

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # load modisco
    mr = ModiscoResult(f"{modisco_dir}/modisco.h5")

    if contribsf is not None:
        d = contribsf
    else:
        d = ContribFile.from_modisco_dir(modisco_dir)
        logger.info("Loading the contribution scores")
        d.cache()  # load all

    thr_one_hot = d.get_seq()
    # thr_hypothetical_contribs
    tracks = d.get_profiles()
    thr_hypothetical_contribs = dict()
    thr_contrib_scores = dict()
    # TODO - generalize this
    thr_hypothetical_contribs['weighted'] = d.get_hyp_contrib()
    thr_contrib_scores['weighted'] = d.get_contrib()

    tasks = d.get_tasks()

    # Count contribution (if it exists)
    if d.contains_contrib_score("counts/pre-act"):
        count_contrib_score = "counts/pre-act"
        thr_hypothetical_contribs['count'] = d.get_hyp_contrib(contrib_score=count_contrib_score)
        thr_contrib_scores['count'] = d.get_contrib(contrib_score=count_contrib_score)
    elif d.contains_contrib_score("count"):
        count_contrib_score = "count"
        thr_hypothetical_contribs['count'] = d.get_hyp_contrib(contrib_score=count_contrib_score)
        thr_contrib_scores['count'] = d.get_contrib(contrib_score=count_contrib_score)
    else:
        # Don't do anything
        pass

    thr_hypothetical_contribs = OrderedDict(flatten(thr_hypothetical_contribs, separator='/'))
    thr_contrib_scores = OrderedDict(flatten(thr_contrib_scores, separator='/'))
    # -------------------------------------------------

    all_seqlets = mr.seqlets()
    all_patterns = mr.patterns()
    if len(all_patterns) == 0:
        print("No patterns found")
        return

    # 1. Plots with tracks and contrib scores
    print("Writing results for contribution scores")
    plot_profiles(all_seqlets,
                  thr_one_hot,
                  tracks=tracks,
                  contribution_scores=thr_contrib_scores,
                  legend=False,
                  flip_neg=True,
                  rotate_y=0,
                  seq_height=.5,
                  patterns=all_patterns,
                  n_bootstrap=100,
                  fpath_template=str(output_dir / "{pattern}/agg_profile_contribcores"),
                  mkdir=True,
                  figsize=figsize)

    # 2. Plots only with hypothetical contrib scores
    print("Writing results for hypothetical contribution scores")
    plot_profiles(all_seqlets,
                  thr_one_hot,
                  tracks={},
                  contribution_scores=thr_hypothetical_contribs,
                  legend=False,
                  flip_neg=True,
                  rotate_y=0,
                  seq_height=1,
                  patterns=all_patterns,
                  n_bootstrap=100,
                  fpath_template=str(output_dir / "{pattern}/agg_profile_hypcontribscores"),
                  figsize=figsize)

    print("Plotting heatmaps")
    for pattern in tqdm(all_patterns):
        write_heatmap_pngs(all_seqlets[pattern],
                           d,
                           tasks,
                           pattern,
                           output_dir=str(output_dir / pattern))

    mr.close()


def modisco_centroid_seqlet_matches(modisco_dir, contrib_scores, output_dir,
                                    trim_frac=0.08, n_jobs=1,
                                    contribsf=None):
    """Write pattern matches to .csv
    """
    from bpnet.modisco.table import ModiscoData
    assert os.path.exists(output_dir)
    add_file_logging(output_dir, logger, 'centroid-seqlet-matches')

    logger.info("Loading required data")
    data = ModiscoData.load(modisco_dir, contrib_scores, contribsf=contribsf)

    logger.info("Generating the table")
    df = data.get_centroid_seqlet_matches(trim_frac=trim_frac, n_jobs=n_jobs)
    df.to_csv(os.path.join(output_dir, 'centroid_seqlet_matches.csv'))


def cwm_scan(modisco_dir,
             output_file,
             trim_frac=0.08,
             contrib_scores=None,
             contribution=None,
             ignore_filter=False,
             n_jobs=20):
    """Modisco score instances

    Args:
      modisco_dir: modisco directory - used to obtain centroid_seqlet_matches.csv and modisco.h5
      output_file: output file path for the tsv file. If the suffix is
        tsv.gz, then also gzip the file
      trim_frac: how much to trim the pattern when scanning
      contrib_scores: hdf5 file of contribution scores (contains `contribution` score)
        if None, then load the default contribution scores from modisco
      contribution: which contribution scores to use
      n_jobs: number of parallel jobs to use

    Writes a gzipped tsv file(tsv.gz)
    """
    add_file_logging(os.path.dirname(output_file), logger, 'modisco-score2')
    modisco_dir = Path(modisco_dir)
    modisco_kwargs = read_json(f"{modisco_dir}/modisco-run.kwargs.json")
    if contribution is None:
        # HACK - fix that to match the bpnet_modisco_run command
        contribution = modisco_kwargs['contrib_wildcard'].split(",")[0].split("/", maxsplit=1)[1]

    # Centroid matches
    cm_path = modisco_dir / 'centroid_seqlet_matches.csv'
    if not cm_path.exists():
        logger.info(f"Generating centroid matches to {cm_path.resolve()}")
        modisco_centroid_seqlet_matches(modisco_dir,
                                        contrib_scores,
                                        modisco_dir,
                                        trim_frac=trim_frac,
                                        n_jobs=n_jobs)
    logger.info(f"Loading centroid matches from {cm_path.resolve()}")
    dfm_norm = pd.read_csv(cm_path)

    mr = ModiscoResult(modisco_dir / "modisco.h5")
    mr.open()
    tasks = mr.tasks()

    # HACK prune the tasks of contribution (in case it's present)
    tasks = [t.replace(f"/{contribution}", "") for t in tasks]

    logger.info(f"Using tasks: {tasks}")

    # TODO - use the same subsetting as before
    if contrib_scores is not None:
        logger.info(f"Loading the contribution scores from: {contrib_scores}")
        contrib = ContribFile(contrib_scores, default_contrib_score=contribution)
    else:
        contrib = ContribFile.from_modisco_dir(modisco_dir, ignore_include_samples=ignore_filter)

    seq, contrib, hyp_contrib, profile, ranges = contrib.get_all()

    logger.info("Scanning for patterns")
    dfl = []
    for pattern_name in tqdm(mr.patterns()):
        pattern = mr.get_pattern(pattern_name).trim_seq_ic(trim_frac)
        match, contribution = pattern.scan_contribution(contrib, hyp_contrib, tasks,
                                                        n_jobs=n_jobs, verbose=False)
        seq_match = pattern.scan_seq(seq, n_jobs=n_jobs, verbose=False)
        dfm = pattern.get_instances(tasks, match, contribution, seq_match,
                                    norm_df=dfm_norm[dfm_norm.pattern == pattern_name],
                                    verbose=False, plot=False)
        dfl.append(dfm)

    logger.info("Merging")
    # merge and write the results
    dfp = pd.concat(dfl)

    # append the ranges
    logger.info("Append ranges")
    ranges.columns = ["example_" + v for v in ranges.columns]
    dfp = dfp.merge(ranges, on="example_idx", how='left')

    logger.info("Table info")
    dfp.info()
    logger.info(f"Writing the resuling pd.DataFrame of shape {dfp.shape} to {output_file}")
    # write to a parquet file
    dfp.to_parquet(output_file, partition_on=['pattern'], engine='fastparquet')
    logger.info("Done!")
    # except:
    #    import pdb
    #    pdb.set_trace()
    # dfp.to_csv(output_file, index=False, sep='\t', compression='gzip')


def modisco_report(modisco_dir, output_dir):
    render_ipynb(os.path.join(this_path, "../templates/modisco.ipynb"),
                 os.path.join(output_dir, "results.ipynb"),
                 params=dict(modisco_dir=modisco_dir))


def modisco_cluster_patterns(modisco_dir, output_dir):
    render_ipynb(os.path.join(this_path, "../modisco/cluster-patterns.ipynb"),
                 os.path.join(output_dir, "cluster-patterns.ipynb"),
                 params=dict(modisco_dir=str(modisco_dir),
                             output_dir=str(output_dir)))


def modisco2bed(modisco_dir, output_dir, trim_frac=0.08):
    from pybedtools import Interval
    from bpnet.modisco.results import ModiscoResult
    add_file_logging(output_dir, logger, 'modisco2bed')
    ranges = load_ranges(modisco_dir)
    example_intervals = [Interval(row.chrom, row.start, row.end)
                         for i, row in ranges.iterrows()]

    r = ModiscoResult(os.path.join(modisco_dir, "modisco.h5"))
    r.export_seqlets_bed(output_dir,
                         example_intervals=example_intervals,
                         position='absolute',
                         trim_frac=trim_frac)
    r.close()


def modisco_table(modisco_dir, contrib_scores, output_dir, report_url=None, contribsf=None):
    """Write the pattern table to as .html and .csv
    """
    plt.switch_backend('agg')
    from bpnet.modisco.table import ModiscoData, modisco_table, write_modisco_table
    from bpnet.modisco.motif_clustering import hirearchically_reorder_table
    add_file_logging(output_dir, logger, 'modisco-table')
    print("Loading required data")
    data = ModiscoData.load(modisco_dir, contrib_scores, contribsf=contribsf)

    print("Generating the table")
    df = modisco_table(data)

    print("Writing the results")
    write_modisco_table(df, output_dir, report_url, 'pattern_table')

    print("Writing clustered table")
    write_modisco_table(hirearchically_reorder_table(df, data.tasks),
                        output_dir, report_url, 'pattern_table.sorted')

    print("Writing footprints")
    profiles = OrderedDict([(pattern, {task: data.get_profile_wide(pattern, task).mean(axis=0)
                                       for task in data.tasks})
                            for pattern in data.mr.patterns()])
    write_pkl(profiles, Path(output_dir) / 'footprints.pkl')
    print("Done!")


def modisco_centroid_seqlet_matches2(modisco_dir, contrib_scores, output_dir=None, trim_frac=0.08, n_jobs=1, data_class='profile'):
    """Write pattern matches to .csv
    """
    from bpnet.modisco.table import ModiscoData, ModiscoDataSingleBinary
    if(output_dir is None):
        output_dir = modisco_dir
    add_file_logging(output_dir, logger, 'centroid-seqlet-matches')

    logger.info("Loading required data")
    if(data_class == 'profile'):
        data = ModiscoData.load(modisco_dir, contrib_scores)
    else:
        data = ModiscoDataSingleBinary.load(modisco_dir)

    logger.info("Generating the table")
    df = data.get_centroid_seqlet_matches(trim_frac=trim_frac, n_jobs=n_jobs)

    df.to_csv(os.path.join(output_dir, 'centroid_seqlet_matches.csv'))

    return None


def modisco_enrich_patterns(patterns_pkl_file, modisco_dir, output_file, contribsf=None):
    """Add stacked_seqlet_contrib to pattern `attrs`

    Args:
      patterns_pkl: patterns.pkl file path
      modisco_dir: modisco directory containing
      output_file: output file path for patterns.pkl
    """
    from bpnet.utils import read_pkl, write_pkl
    from bpnet.cli.contrib import ContribFile

    logger.info("Loading patterns")
    modisco_dir = Path(modisco_dir)
    patterns = read_pkl(patterns_pkl_file)

    mr = ModiscoResult(modisco_dir / 'modisco.h5')
    mr.open()

    if contribsf is None:
        contrib_file = ContribFile.from_modisco_dir(modisco_dir)
        logger.info("Loading ContribFile into memory")
        contrib_file.cache()
    else:
        logger.info("Using the provided ContribFile")
        contrib_file = contribsf

    logger.info("Extracting profile and contribution scores")
    extended_patterns = []
    for p in tqdm(patterns):
        p = p.copy()
        profile_width = p.len_profile()
        # get the shifted seqlets
        seqlets = [s.pattern_align(**p.attrs['align']) for s in mr._get_seqlets(p.name)]

        # keep only valid seqlets
        valid_seqlets = [s for s in seqlets
                         if s.valid_resize(profile_width, contrib_file.get_seqlen() + 1)]
        # extract the contribution scores
        p.attrs['stacked_seqlet_contrib'] = contrib_file.extract(valid_seqlets, profile_width=profile_width)

        p.attrs['n_seqlets'] = mr.n_seqlets(*p.name.split("/"))
        extended_patterns.append(p)

    write_pkl(extended_patterns, output_file)


def modisco_export_patterns(modisco_dir, output_file, contribsf=None):
    """Export patterns to a pkl file. Don't cluster them

    Adds `stacked_seqlet_contrib` and `n_seqlets` to pattern `attrs`

    Args:
      patterns_pkl: patterns.pkl file path
      modisco_dir: modisco directory containing
      output_file: output file path for patterns.pkl
    """
    from bpnet.cli.contrib import ContribFile

    logger.info("Loading patterns")
    modisco_dir = Path(modisco_dir)

    mr = ModiscoResult(modisco_dir / 'modisco.h5')
    mr.open()
    patterns = [mr.get_pattern(pname)
                for pname in mr.patterns()]

    if contribsf is None:
        contrib_file = ContribFile.from_modisco_dir(modisco_dir)
        logger.info("Loading ContribFile into memory")
        contrib_file.cache()
    else:
        logger.info("Using the provided ContribFile")
        contrib_file = contribsf

    logger.info("Extracting profile and contribution scores")
    extended_patterns = []
    for p in tqdm(patterns):
        p = p.copy()

        # get the shifted seqlets
        valid_seqlets = mr._get_seqlets(p.name)

        # extract the contribution scores
        sti = contrib_file.extract(valid_seqlets, profile_width=None)
        sti.dfi = mr.get_seqlet_intervals(p.name, as_df=True)
        p.attrs['stacked_seqlet_contrib'] = sti
        p.attrs['n_seqlets'] = mr.n_seqlets(*p.name.split("/"))
        extended_patterns.append(p)

    write_pkl(extended_patterns, output_file)


def modisco_report_all(modisco_dir, trim_frac=0.08, n_jobs=20, scan_instances=False, force=False):
    """Compute all the results for modisco. Runs:
    - modisco_plot
    - modisco_report
    - modisco_table
    - modisco_centroid_seqlet_matches
    - cwm_scan
    - modisco2bed
    - modisco_instances_to_bed

    Args:
      modisco_dir: directory path `output_dir` in `bpnet.cli.modisco.modisco_run`
        contains: modisco.h5, strand_distances.h5, modisco-run.kwargs.json
      trim_frac: how much to trim the pattern
      n_jobs: number of parallel jobs to use
      force: if True, commands will be re-run regardless of whether whey have already
        been computed

    Note:
      All the sub-commands are only executed if they have not been ran before. Use --force override this.
      Whether the commands have been run before is deterimined by checking if the following file exists:
        `{modisco_dir}/.modisco_report_all/{command}.done`.
    """
    plt.switch_backend('agg')
    from bpnet.utils import ConditionalRun

    modisco_dir = Path(modisco_dir)
    # figure out the contribution scores used
    kwargs = read_json(modisco_dir / "modisco-run.kwargs.json")
    contrib_scores = kwargs["contrib_file"]

    mr = ModiscoResult(f"{modisco_dir}/modisco.h5")
    mr.open()
    all_patterns = mr.patterns()
    mr.close()
    if len(all_patterns) == 0:
        print("No patterns found.")
        # Touch results.html for snakemake
        open(modisco_dir / 'results.html', 'a').close()
        open(modisco_dir / 'seqlets/scored_regions.bed', 'a').close()
        return

    # class determining whether to run the command or not (poor-man's snakemake)
    cr = ConditionalRun("modisco_report_all", None, modisco_dir, force=force)

    sync = []
    # --------------------------------------------
    if (not cr.set_cmd('modisco_plot').done()
        or not cr.set_cmd('modisco_cluster_patterns').done()
            or not cr.set_cmd('modisco_enrich_patterns').done()):
        # load ContribFile and pass it to all the functions
        logger.info("Loading ContribFile")
        contribsf = ContribFile.from_modisco_dir(modisco_dir)
        contribsf.cache()
    else:
        contribsf = None
    # --------------------------------------------
    # Basic reports
    if not cr.set_cmd('modisco_plot').done():
        modisco_plot(modisco_dir,
                     modisco_dir / 'plots',
                     figsize=(10, 10), contribsf=contribsf)
        cr.write()
    sync.append("plots")

    if not cr.set_cmd('modisco_report').done():
        modisco_report(str(modisco_dir), str(modisco_dir))
        cr.write()
    sync.append("results.html")

    if not cr.set_cmd('modisco_table').done():
        modisco_table(modisco_dir, contrib_scores, modisco_dir, report_url=None, contribsf=contribsf)
        cr.write()
    sync.append("footprints.pkl")
    sync.append("pattern_table.*")

    if not cr.set_cmd('modisco_cluster_patterns').done():
        modisco_cluster_patterns(modisco_dir, modisco_dir)
        cr.write()
    sync.append("patterns.pkl")
    sync.append("cluster-patterns.*")
    sync.append("motif_clustering")

    if not cr.set_cmd('modisco_enrich_patterns').done():
        modisco_enrich_patterns(modisco_dir / 'patterns.pkl',
                                modisco_dir,
                                modisco_dir / 'patterns.pkl', contribsf=contribsf)
        cr.write()
    # sync.append("patterns.pkl")

    # TODO - run modisco align
    # - [ ] add the motif clustering step (as ipynb) and export the aligned tables
    #   - save the final table as a result to CSV (ready to be imported in excel)
    # --------------------------------------------
    # Finding new instances
    if scan_instances:
        if not cr.set_cmd('modisco_centroid_seqlet_matches').done():
            modisco_centroid_seqlet_matches(modisco_dir, contrib_scores, modisco_dir,
                                            trim_frac=trim_frac,
                                            n_jobs=n_jobs,
                                            contribsf=contribsf)
            cr.write()

        # TODO - this would not work with the per-TF contribution score file....
        if not cr.set_cmd('cwm_scan').done():
            cwm_scan(modisco_dir,
                     modisco_dir / 'instances.parq',
                     trim_frac=trim_frac,
                     contrib_scores=None,  # Use the default one
                     contribution=None,  # Use the default one
                     n_jobs=n_jobs)
            cr.write()
    # TODO - update the pattern table -> compute the fraction of other motifs etc
    # --------------------------------------------
    # Export bed-files and bigwigs

    # Seqlets
    if not cr.set_cmd('modisco2bed').done():
        modisco2bed(str(modisco_dir), str(modisco_dir / 'seqlets'), trim_frac=trim_frac)
        cr.write()
    sync.append("seqlets")

    # Scanned instances
    # if not cr.set_cmd('modisco_instances_to_bed').done():
    #     modisco_instances_to_bed(str(modisco_dir / 'modisco.h5'),
    #                              instances_parq=str(modisco_dir / 'instances.parq'),
    #                              contrib_score_h5=contrib_scores,
    #                              output_dir=str(modisco_dir / 'instances_bed/'),
    #                              )
    #     cr.write()
    # sync.append("instances_bed")

    # print the rsync command to run in order to sync the output
    # directories to the webserver
    logger.info("Run the following command to sync files to the webserver")
    dirs = " ".join(sync)
    print(f"rsync -av --progress {dirs} <output_dir>/")
