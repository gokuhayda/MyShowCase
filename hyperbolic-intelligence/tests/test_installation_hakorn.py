#!/usr/bin/env python3
"""
Script de Teste - CGT Minimal H-AKORN
======================================

Verifica se a instalação está funcionando corretamente.
"""

import sys
import importlib.util

def test_imports():
    """Testa se todos os imports necessários funcionam."""
    print("🔍 Testando imports...\n")
    
    tests = [
        ("cgt", "Pacote CGT"),
        ("cgt.psi_extensions", "PSI Extensions"),
        ("cgt.psi_extensions.visualization", "Módulo de Visualização"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"✅ {description:<30} OK")
                passed += 1
            else:
                print(f"❌ {description:<30} NÃO ENCONTRADO")
                failed += 1
        except Exception as e:
            print(f"❌ {description:<30} ERRO: {e}")
            failed += 1
    
    return passed, failed


def test_visualization_imports():
    """Testa imports específicos de visualização."""
    print("\n🎨 Testando componentes de visualização...\n")
    
    try:
        from cgt.psi_extensions.visualization import (
            run_hakorn_demo,
            run_realtime_demo,
            HAKORNSimulator,
            MTEB_DATASETS,
            record_hakorn_video,
        )
        
        print(f"✅ run_hakorn_demo         OK")
        print(f"✅ run_realtime_demo       OK")
        print(f"✅ HAKORNSimulator         OK")
        print(f"✅ MTEB_DATASETS           OK ({len(MTEB_DATASETS)} datasets)")
        print(f"✅ record_hakorn_video     OK")
        
        return True, len(MTEB_DATASETS)
    except Exception as e:
        print(f"❌ ERRO ao importar: {e}")
        return False, 0


def test_datasets():
    """Testa se os datasets estão disponíveis."""
    print("\n📊 Testando datasets MTEB...\n")
    
    try:
        from cgt.psi_extensions.visualization import MTEB_DATASETS
        
        # Agrupar por tipo
        sts_datasets = []
        reranking_datasets = []
        clustering_datasets = []
        
        for name, config in MTEB_DATASETS.items():
            dataset_type = config[-1]
            if dataset_type == 'sts':
                sts_datasets.append(name)
            elif dataset_type == 'reranking':
                reranking_datasets.append(name)
            elif dataset_type == 'clustering':
                clustering_datasets.append(name)
        
        print(f"📈 STS Datasets: {len(sts_datasets)}")
        for ds in sorted(sts_datasets):
            print(f"   • {ds}")
        
        print(f"\n📈 Reranking Datasets: {len(reranking_datasets)}")
        for ds in sorted(reranking_datasets):
            print(f"   • {ds}")
        
        print(f"\n📈 Clustering Datasets: {len(clustering_datasets)}")
        for ds in sorted(clustering_datasets):
            print(f"   • {ds}")
        
        total = len(sts_datasets) + len(reranking_datasets) + len(clustering_datasets)
        print(f"\n✅ Total: {total} datasets disponíveis")
        
        return True, total
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False, 0


def test_dependencies():
    """Testa se dependências principais estão instaladas."""
    print("\n📦 Testando dependências...\n")
    
    deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
    ]
    
    optional_deps = [
        ("sentence_transformers", "Sentence Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("ot", "POT (Optimal Transport)"),
        ("cv2", "OpenCV"),
    ]
    
    passed = 0
    failed = 0
    
    print("Dependências obrigatórias:")
    for module_name, description in deps:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"  ✅ {description}")
                passed += 1
            else:
                print(f"  ❌ {description} - NÃO INSTALADO")
                failed += 1
        except:
            print(f"  ❌ {description} - ERRO")
            failed += 1
    
    print("\nDependências opcionais:")
    for module_name, description in optional_deps:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"  ✅ {description}")
            else:
                print(f"  ⚠️ {description} - NÃO INSTALADO (opcional)")
        except:
            print(f"  ⚠️ {description} - ERRO (opcional)")
    
    return passed, failed


def main():
    """Executa todos os testes."""
    print("=" * 70)
    print("🧪 TESTE DE INSTALAÇÃO - CGT Minimal H-AKORN")
    print("=" * 70)
    
    # Teste 1: Imports básicos
    passed1, failed1 = test_imports()
    
    # Teste 2: Imports de visualização
    success2, num_datasets = test_visualization_imports()
    
    # Teste 3: Datasets
    success3, total_datasets = test_datasets()
    
    # Teste 4: Dependências
    passed4, failed4 = test_dependencies()
    
    # Resumo
    print("\n" + "=" * 70)
    print("📊 RESUMO DOS TESTES")
    print("=" * 70)
    
    print(f"\n✅ Imports básicos:       {passed1} OK, {failed1} falhas")
    print(f"{'✅' if success2 else '❌'} Visualização:         {'OK' if success2 else 'FALHA'}")
    print(f"{'✅' if success3 else '❌'} Datasets MTEB:        {total_datasets} disponíveis")
    print(f"✅ Dependências:          {passed4} instaladas, {failed4} faltando")
    
    # Veredicto final
    all_passed = (failed1 == 0 and success2 and success3 and failed4 == 0)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 SUCESSO! Instalação funcionando perfeitamente!")
        print("=" * 70)
        print("\n📓 Próximo passo: Abra o notebook hakorn_physics_of_meaning.ipynb")
        print("\nExemplo de uso:")
        print("  from cgt.psi_extensions.visualization import run_realtime_demo")
        print("  anim, sim = run_realtime_demo('STSBenchmark', max_samples=50)")
        return 0
    else:
        print("⚠️ ATENÇÃO! Alguns testes falharam.")
        print("=" * 70)
        if failed1 > 0:
            print("\n❌ Problema: Módulos CGT não encontrados")
            print("   Solução: Certifique-se que o pacote está no PYTHONPATH")
            print("   Execute: export PYTHONPATH=/caminho/para/cgt_minimal_hakorn/src:$PYTHONPATH")
        if not success2:
            print("\n❌ Problema: Módulos de visualização não importam")
            print("   Solução: Verifique se os arquivos hakorn_*.py existem")
        if failed4 > 0:
            print("\n❌ Problema: Dependências faltando")
            print("   Solução: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
