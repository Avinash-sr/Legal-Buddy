import 'package:flutter/material.dart';


class CardList extends StatelessWidget {
  final List<Map<String, dynamic>> legalActs = [
    {
      "id": 7,
      "code": "IPC-498A",
      "name": "IPC-498A",
      "description": "Deals with cruelty to a married woman by her husband or his relatives",
      "helpline": "1091"
    },
    {
      "id": 17,
      "code": "IPC 326",
      "name": "IPC 326",
      "description": "Voluntarily causing grievous hurt by dangerous weapons or means. Imprisonment for Life or 10 Years + Fine",
      "helpline": "100"
    },
    {
      "id": 6,
      "code": "IPC-354A",
      "name": "IPC-354A",
      "description": "Sexual Harassment and Punishment for Sexual Harassment",
      "helpline": "7827170170"
    },
    {
      "id": 20,
      "code": "DVA",
      "name": "Domestic Violence Act 2005",
      "description": "Protection of women from domestic violence",
      "helpline": "1091"
    },
    {
      "id": 19,
      "code": "DPA",
      "name": "Dowry Protection Act 1984",
      "description": "No dowry allowed",
      "helpline": "100"
    }
  ];

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: legalActs.length,
      itemBuilder: (context, index) {
        return LegalActCard(legalAct: legalActs[index]);
      },
    );
  }
}

class LegalActCard extends StatelessWidget {
  final Map<String, dynamic> legalAct;

  LegalActCard({required this.legalAct});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: EdgeInsets.all(16.0),
      child: ListTile(
        title: Text(legalAct['name']),
        subtitle: Text(legalAct['description']),
        trailing: Text('Helpline: ${legalAct['helpline']}'),
      ),
    );
  }
}