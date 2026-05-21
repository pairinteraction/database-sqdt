# Tool for generating the SQDT database tables in the cloud

Create the database including tables of states and matrix elements calculated with single-channel quantum defect theory.
Database tables are available through [GitHub Releases](https://github.com/pairinteraction/database-sqdt/releases).

## Generate new tables locally
Run the script to generate the tables via
```bash
uv run generate_database <species> --n-max <n-max> --directory <directory>
```
or for the misc wigner table via
```bash
uv run generate_database misc --f-max <f-max> --directory <directory>
```

## Generate a new release
To generate a new release with all the important tables, simply create and push a new annotated tag with a tag name of the form `v*.*`.
This will run the `generate_database.yml` workflow, where first all tables are created,
and then in addition the zipped versions of the tables are uploaded to a new release with name `v*.*`.
The release is created in draft mode, so you can double-check, that all database tables are included and optionally add a release text.
Once you are happy with the release draft, don't forget to publish the release.

## Misc

### Profiling / Benchmarking
To check the performance of this tool and to look for bottlenecks in the code, you can use [py-spy](https://github.com/benfred/py-spy) to profile the code.
To do so simply run the following command:

```bash
uv run py-spy record -o profiling.svg -- generate_database <species> --n-max <n-max>
```
