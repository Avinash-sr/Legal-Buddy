// ignore_for_file: prefer_const_constructors, prefer_const_literals_to_create_immutables, must_be_immutable

import 'package:flutter/material.dart';
import 'legal_consultant.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {

   int _selectedIndex = 0;

  // final List<Widget> _pages = [
  //   MyHomePage(),
  //   Text('History Page'),
  //   Text('Profile Page'),
  // ];

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        bottomNavigationBar:Padding(
          padding: const EdgeInsets.all(10.0),
          child: Container(
          decoration: BoxDecoration(
            color: Color(0xFF161616),
            borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
          ),
          child: BottomNavigationBar(
            backgroundColor: Colors.transparent,
            selectedItemColor: Colors.white,
            unselectedItemColor: Colors.grey,
            currentIndex: _selectedIndex,
            onTap: _onItemTapped,
            items: [
              BottomNavigationBarItem(
                icon: Icon(Icons.home),
                label: 'Home',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.history),
                label: 'History',
              ),
              BottomNavigationBarItem(
                icon: Icon(Icons.person),
                label: 'Profile',
              ),
            ],
          ),
              ),
        ),
        backgroundColor: const Color(0xFF2F3B3C),
        body: SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 10.0, vertical: 10),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      "Hello, \nAvinash Kumar",
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold),
                    ),
                    CircleAvatar(
                      radius: 30,
                      backgroundImage: AssetImage("images/logo1.png"),
                    )
                  ],
                ),
              ),
              SizedBox(height: 20,),
              SizedBox(
                width: MediaQuery.of(context).size.width / 1.2,
                height: 100,
                child: GestureDetector(
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => LegalConsultantScreen(),));
                  },
                  child: Card(
                    color: Colors.white,
                    child: Center(
                      child: Text(
                        "New Query",
                        style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(
                height: 20,
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Explore more',
                      style: TextStyle(fontSize: 20, color: Colors.white),
                    ),
                    Container(
                        height: 40,
                        width: 40,
                        decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(100),
                            color: Colors.white),
                        child: Icon(
                          Icons.arrow_forward,
                          color: Colors.black,
                        ))
                  ],
                ),
              ),
              SingleChildScrollView(
                // scrollDirection: Axis.vertical,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(10),
                      child: FeatureContainer(text: "Case Studies", icon: Icons.file_copy,description: "Real life past registered cases according to your query.",),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(10.0),
                      child: FeatureContainer(text: "Documentation", icon: Icons.document_scanner_outlined, description: "List of documents required for the mentioned query.",),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(10.0),
                      child: FeatureContainer(text: "Legal Consultant", icon: Icons.file_copy, description: "Consult user according to the query entered.",),
                    ),
                  ],
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }
}


class FeatureContainer extends StatelessWidget {
  String? text;
  String? description;
  IconData? icon;
  FeatureContainer({super.key, this.icon, this.text, this.description});

  @override
  Widget build(BuildContext context) {
    return Container(
        height: 100,
        width: MediaQuery.of(context).size.width/1.2,
        decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        color: Color(0xFF616161)
      ),
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Row(children: [
            Icon(icon, color: Colors.white,),
            Text("$text", style: TextStyle(color: Colors.white),),
          ],),
          Text("$description", style: TextStyle(color: Colors.grey),)
        ],
        ),
      ),
    );
  }
}

