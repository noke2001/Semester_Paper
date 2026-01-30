import openml
import os

# 334 - 337
suite_ids = [334, 335, 336, 337]
cache_root = "/cluster/home/pdamota/openml_cache"

output_file = "download_all.sh"
seen_datasets = set()

print(f"Preparing to generate download script for suites: {suite_ids}")

with open(output_file, "w") as f:
    f.write("#!/bin/bash\n")
    # Create the cache root structure
    f.write(f"mkdir -p {cache_root}\n")
    f.write("echo 'Starting Bulk Download for all Suites...'\n")

    for s_id in suite_ids:
        print(f"--> Fetching metadata for Suite {s_id}...")
        try:
            suite = openml.study.get_suite(s_id)
        except Exception as e:
            print(f"❌ Failed to get info for suite {s_id}: {e}")
            continue

        f.write(f"\n# --- Processing Suite {s_id} ---\n")

        for task_id in suite.tasks:
            try:
                # 1. Get Task Metadata (Lightweight)
                task = openml.tasks.get_task(task_id, download_data=False)
                dataset_id = task.dataset_id
                
                # 2. Skip duplicates
                if dataset_id in seen_datasets:
                    continue
                seen_datasets.add(dataset_id)

                # 3. Get Dataset Metadata (Contains the URL)
                ds = openml.datasets.get_dataset(dataset_id, download_data=False)

                # 4. Extract URL safely
                # 'url' attribute contains the direct link to the ARFF file
                if hasattr(ds, 'url') and ds.url:
                    download_url = ds.url
                else:
                    print(f"   ⚠️ Warning: Dataset {dataset_id} has no 'url' attribute. Skipping.")
                    continue

                # 5. Construct Target Path (Standard OpenML Cache Structure)
                target_dir = f"{cache_root}/org/openml/www/datasets/{dataset_id}"
                target_file = f"{target_dir}/dataset.arff"

                # 6. Write Command
                f.write(f"mkdir -p {target_dir}\n")
                # -nc = No Clobber (skip if exists)
                # -q = Quiet (less spam)
                f.write(f"wget -nc -q --show-progress -O {target_file} '{download_url}'\n")
                f.write(f"echo 'Downloaded Dataset {dataset_id} ({ds.name})'\n")

            except Exception as e:
                # This catches errors for specific tasks without stopping the whole script
                print(f"   Skipping task {task_id}: {e}")

print(f"\n✅ DONE! Generated '{output_file}' covering {len(seen_datasets)} unique datasets.")
print("Now run this on the login node: bash download_all.sh")