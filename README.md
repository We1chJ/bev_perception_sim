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
