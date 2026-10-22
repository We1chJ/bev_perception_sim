# bev_perception_sim

Bird's-eye-view (BEV) perception simulation built on the Waymo Open Dataset.

## Setup

1. Register for the [Waymo Open Dataset](https://waymo.com/open/) with a Google account.
2. Authenticate the Google Cloud CLI with that same account:

   ```bash
   gcloud auth login
   ```

3. Verify bucket read access:

   ```bash
   gcloud storage ls gs://waymo_open_dataset_v_1_4_2/
   ```

   If it lists directories (such as `individual_files/`, `all_files/`, or `v1.4.2/`), your permissions are active.

## Download test segments

Copy one or two validation segments (about 0.9 GB each) into `data/waymo/raw/`, which is ignored by git:

```bash
mkdir -p ./data/waymo/raw
CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0 gcloud storage cp \
  gs://waymo_open_dataset_v_1_4_2/individual_files/validation/segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord \
  gs://waymo_open_dataset_v_1_4_2/individual_files/validation/segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord \
  ./data/waymo/raw/
```

To browse other segments with their sizes:

```bash
gcloud storage ls -l "gs://waymo_open_dataset_v_1_4_2/individual_files/validation/segment-102*"
```

**Why sliced downloads are disabled:** with gcloud's default sliced (parallel) download, the copy finishes but writes an empty file, fails its checksum check (`destination hash AAAAAA==`), and deletes the result. Setting the threshold to `0` downloads each file in one stream, which works. To make this permanent:

```bash
gcloud config set storage/sliced_object_download_threshold 0
```
