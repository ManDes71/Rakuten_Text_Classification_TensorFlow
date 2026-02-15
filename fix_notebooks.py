#!/usr/bin/env python3
"""
Script pour nettoyer les métadonnées de widgets des notebooks Jupyter
Résout l'erreur GitHub: "the 'state' key is missing from 'metadata.widgets'"
"""
import json
import glob
import sys

def clean_notebook_metadata(notebook_path):
    """Supprime les métadonnées widgets problématiques d'un notebook"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Vérifie si les métadonnées widgets existent
        if 'widgets' in notebook.get('metadata', {}):
            # Supprime les métadonnées widgets
            del notebook['metadata']['widgets']
            
            # Sauvegarde le notebook nettoyé
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            
            return True
        return False
    except Exception as e:
        print(f"✗ Erreur avec {notebook_path}: {e}")
        return None

def main():
    notebooks = glob.glob("notebooks/*.ipynb")
    
    if not notebooks:
        print("Aucun notebook trouvé dans le dossier notebooks/")
        sys.exit(1)
    
    print(f"Nettoyage de {len(notebooks)} notebooks...\n")
    
    cleaned = 0
    skipped = 0
    errors = 0
    
    for nb_path in sorted(notebooks):
        nb_name = nb_path.split('/')[-1]
        result = clean_notebook_metadata(nb_path)
        
        if result is True:
            print(f"✓ {nb_name} - nettoyé")
            cleaned += 1
        elif result is False:
            print(f"○ {nb_name} - déjà propre")
            skipped += 1
        else:
            errors += 1
    
    print(f"\n{'='*60}")
    print(f"Résumé:")
    print(f"  • Nettoyés: {cleaned}")
    print(f"  • Déjà propres: {skipped}")
    print(f"  • Erreurs: {errors}")
    print(f"{'='*60}")
    
    if cleaned > 0:
        print("\n✓ Les notebooks peuvent maintenant être affichés sur GitHub!")

if __name__ == "__main__":
    main()
