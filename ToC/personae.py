prompt = """In einem deutschen Krankenhaus, in dem ein Portkatheter gelegt wird, arbeiten drei Personen im OP zusammen: ein Radiologe, der die Operation durchführt, ein medizinischer Assistent, der den Radiologen bei der Bedienung der Geräte, insbesondere des Röntgengeräts, unterstützt, und der Patient, der operiert wird. 

Natürlich haben alle diese Personen unterschiedliche Persönlichkeiten, Dialekte oder Hintergründe. Ihre Aufgabe ist es, 3 verschiedene Personas für jede dieser Personen zu erstellen. 

Geben Sie für jede Persona an, wie sie spricht. Diese Personas sollen in einer textbasierten Anwendung verwendet werden. Daher sollten die Unterschiede in ihrem Gesprächsstil in den Transkriptionen ihrer Gespräche, also im Text, sichtbar sein.

Geben Sie für den Patienten die relevante Krankengeschichte an, wenn sie vorhanden ist.

Geben Sie Ihre Antwort im csv-Format wie unten:

- Radiologe
Index,Persona

- Assistentin
Index,Persona

- Patient
Index,Persona,Medizinische_Geschichte
"""