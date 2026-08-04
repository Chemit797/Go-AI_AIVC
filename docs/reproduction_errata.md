# Document-to-Data Compatibility Record

The implementation preserves the document baseline method while correcting released-data mismatches and non-executable snippets.

| Document expression | Released-data implementation | Reason | Method changed? |
|---|---|---|---|
| `strain` | `Strains` | Released metadata field name | No |
| `chemical` | `perturbation_no_concentration` | This is the stable perturbation name; bare `pert_id` is not globally unique | No |
| `medium` | `Medium` | Released metadata field name | No |
| `temperature` | `Temperature` | Released metadata field name | No |
| `time` in hours | `pert_time` plus `pert_time_unit` in minutes | Released times are minutes | No |
| `plate` | `Yeast_cell_plate` | Released metadata field name | No |
| `product_id` for control/QC | perturbation name: Water/DMSO/control and Quality Control/QC | `product_id` is not released | No |
| 4,232 output proteins | dynamically derived training-only feature contract | The document count is not valid for the released matrix | No |
| `np.concat` | `np.concatenate` | `np.concat` does not exist | No |
| 32 dimensions from one MD5 digest | counter-extended MD5 bytes | A single MD5 digest has only 16 bytes | No |

The implementation never hard-codes the retained protein count. It records the actual result of the specified `<80%` training missingness rule in every run's feature contract.
