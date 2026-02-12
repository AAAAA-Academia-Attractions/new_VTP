#!/bin/bash

# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets en --keep-empty-query
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets es --keep-empty-query
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets it --keep-empty-query
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets de --keep-empty-query
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets fr --keep-empty-query

python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --keep-empty-query --output-root /root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow --dataset-name llamaindex/vdr-multilingual-test
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets es --keep-empty-query --output-root /root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow_es --dataset-name llamaindex/vdr-multilingual-test
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets it --keep-empty-query --output-root /root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow_it --dataset-name llamaindex/vdr-multilingual-test
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets de --keep-empty-query --output-root /root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow_de --dataset-name llamaindex/vdr-multilingual-test
# python /root/autodl-tmp/new_VTP/src/data_prep/embed_vdr_with_jina_internal.py --subsets fr --keep-empty-query --output-root /root/autodl-tmp/hf/processed/vdr_multilingual_test_internal_arrow_fr --dataset-name llamaindex/vdr-multilingual-test