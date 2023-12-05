// ignore_for_file: prefer_final_fields, unused_field, library_private_types_in_public_api, unnecessary_null_comparison

import 'package:flutter/material.dart';
import 'dart:async';
import 'package:legal_buddy2/utils/ListCard.dart';

class LegalConsultantScreen extends StatefulWidget {
  const LegalConsultantScreen({Key? key}) : super(key: key);

  @override
  _LegalConsultantScreenState createState() => _LegalConsultantScreenState();
}

class _LegalConsultantScreenState extends State<LegalConsultantScreen> {
  TextEditingController _queryController = TextEditingController();
  List<String> _messages = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF2F3B3C),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.only(top: 10.0, left: 20, right: 20),
          child: Column(
            children: [
              Row(
                children: [
                  Padding(
                    padding: const EdgeInsets.only(right: 20.0),
                    child: GestureDetector(
                      onTap: () {
                        Navigator.pop(context);
                      },
                      child: Container(
                        height: 40,
                        width: 40,
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(100),
                            color: Colors.white),
                        child: const Icon(
                          Icons.arrow_back_rounded,
                          color: Colors.black,
                        ),
                      ),
                    ),
                  ),
                  Center(
                    child: Container(
                      child: const Text(
                        "LEGAL CONSULTANT",
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 25,
                            fontWeight: FontWeight.bold,),
                      ),
                    ),
                  ),
                ],
              ),
              Container(
                color: Colors.white,
                width: 10,
              ),
              Expanded(
                child: Center(
                  child: Container(
                    decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(15),
                        color: Colors.white),
                    height: MediaQuery.of(context).size.height / 1.5,
                    child: ListView.builder(
                      itemCount: _messages.length,
                      itemBuilder: (context, index) {
                        return _buildMessage(_messages[index]);
                      },
                    ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Row(
                  children: [
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          color: const Color(0xFF161616),
                        ),
                        child: Padding(
                          padding: const EdgeInsets.only(left: 8.0),
                          child: TextField(
                            style: const TextStyle(color: Colors.white),
                            controller: _queryController,
                            decoration: const InputDecoration(
                              fillColor: Colors.white,
                              border: OutlineInputBorder(
                                borderSide: BorderSide.none,
                              ),
                              hintText: 'Type your query...',
                              hintStyle: TextStyle(color: Colors.white),
                            ),
                          ),
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(
                        Icons.send,
                        color: Colors.white,
                      ),
                      onPressed: () {
                        _sendMessage();
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMessage(String message) {
    return Padding(
      padding: const EdgeInsets.only(left: 10.0, right: 10),
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8.0),
        padding: const EdgeInsets.all(8.0),
        decoration: BoxDecoration(
          color: const Color(0xFF616161),
          borderRadius: BorderRadius.circular(8.0),
        ),
        child: Container(
          decoration: BoxDecoration(borderRadius: BorderRadius.circular(20)),
          child: Row(
            children: [
              const Icon(Icons.person_2_rounded,color: Colors.white,),
              const SizedBox(width: 8,),
              Text(
              message,
              style: const TextStyle(color: Colors.white),
            ),
            ],
          ),
        ),
      ),
    );
  }

  // String? response = '';
  // void show_message(String response) {
  //   for(int i = 0; i < legalActs.length; i++){
  //     response += '$legalActs[]'
  //   }
  // }

  void _sendMessage() {
    String userMessage = _queryController.text.trim();
    if (userMessage.isNotEmpty) {
      setState(() {
        _messages.add('You: $userMessage');
        Future.delayed(const Duration(seconds: 1), () {
          String botResponse = 'Bot: Thankyou! you query has been \nsubmitted successfully.';
          setState(() {
            _messages.add(botResponse);
          });
        });
      });

      _queryController.clear();
    }
  }


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

}
